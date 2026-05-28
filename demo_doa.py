"""Minimal Radar SubspaceNet DoA demo.

Loads the latest trained Radar SubspaceNet model by default, reads one radar
array observation, and prints the DoA estimate against ground truth.
"""

import argparse
from itertools import permutations
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm

import src.data_handler as data_handler
from src.models import SubspaceNet


DEFAULT_MODEL_PATH = (
    Path("data")
    / "weights"
    / "final_models"
    / "Radar_SubspaceNet_M=2_T=200_SNR_0_tau=8_NarrowBand_diff_method=root_music_non-coherent_eta=0_bias=0_sv_noise=0.pt"
)
RADAR_MODEL_GLOB = "Radar_SubspaceNet*diff_method=root_music*.pt"
DEFAULT_DATASET_PATH = (
    Path("data")
    / "datasets"
    / "radar_baseline"
    / "test"
    / "Generic_DataSet_NarrowBand_non-coherent_500_M=2_N=16_T=200_SNR=0_eta=0_sv_noise_var0_bias=0_.h5"
)
DEFAULT_BATCH_DATASET_PATH = (
    Path("data")
    / "datasets"
    / "radar_baseline"
    / "test"
    / "SubspaceNet_DataSet_NarrowBand_non-coherent_500_M=2_N=16_T=200_SNR=0_eta=0_sv_noise_var0_bias=0_.h5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal SubspaceNet DoA estimation demo."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to the trained Radar SubspaceNet state_dict. Defaults to latest Radar_*.pt.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to a dataset containing (X, doa) samples.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index to use from the dataset.",
    )
    parser.add_argument(
        "--batch",
        nargs="?",
        const=500,
        type=int,
        default=None,
        help="Evaluate the first N samples from the dataset. Defaults to 500 when used.",
    )
    parser.add_argument(
        "--batch-dataset-path",
        type=Path,
        default=None,
        help="Path to a precomputed SubspaceNet dataset for batch evaluation.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device.",
    )
    return parser.parse_args()


def resolve_model_path(model_path: Path | None) -> Path:
    if model_path is not None:
        return model_path
    models_dir = DEFAULT_MODEL_PATH.parent
    radar_models = sorted(
        models_dir.glob(RADAR_MODEL_GLOB),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if radar_models:
        return radar_models[0]
    return DEFAULT_MODEL_PATH


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_name)


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def infer_tau(state_dict: dict) -> int:
    conv1_weight = state_dict.get("conv1.weight")
    if conv1_weight is None or conv1_weight.ndim != 4:
        raise RuntimeError("Could not infer tau from model weights.")
    return int(conv1_weight.shape[1])


def best_periodic_match(pred_deg: np.ndarray, truth_deg: np.ndarray):
    best = None
    for candidate in permutations(pred_deg, len(pred_deg)):
        candidate = np.asarray(candidate)
        error = ((candidate - truth_deg + 90.0) % 180.0) - 90.0
        rmse = float(np.sqrt(np.mean(error**2)))
        if best is None or rmse < best[2]:
            best = (candidate, error, rmse)
    return best


def format_values(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:8.3f}" for value in values) + "]"


def load_model(model_path: Path, device: torch.device, num_sources: int):
    state_dict = torch_load(model_path, device)
    tau = infer_tau(state_dict)
    model = SubspaceNet(tau=tau, M=num_sources, diff_method="root_music").to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, tau


def predict_doa(
    model: SubspaceNet, sample_input: torch.Tensor, tau: int, device: torch.device
) -> tuple[np.ndarray, float]:
    if torch.is_complex(sample_input):
        model_input = data_handler.create_autocorrelation_tensor(sample_input, tau)
    else:
        model_input = sample_input
    model_input = model_input.unsqueeze(0).to(device=device, dtype=torch.float32)
    start = perf_counter()
    with torch.no_grad():
        pred_rad, _, _, _ = model(model_input)
    elapsed_ms = (perf_counter() - start) * 1000.0
    pred_deg = np.rad2deg(pred_rad.squeeze(0).detach().cpu().numpy())
    return pred_deg, elapsed_ms


def run_batch(
    dataset: list, model: SubspaceNet, tau: int, device: torch.device, limit: int
) -> None:
    if limit <= 0:
        raise ValueError("--batch must be a positive integer.")

    count = min(limit, len(dataset))
    rmspe_values = []
    inference_times = []

    print(f"Batch samples: {count}")
    for index in tqdm(range(count), desc="Evaluating"):
        X, truth_rad = dataset[index]
        pred_deg, elapsed_ms = predict_doa(model, X, tau, device)
        truth_deg = np.rad2deg(truth_rad.detach().cpu().numpy())
        _, _, rmse = best_periodic_match(pred_deg, truth_deg)
        rmspe_values.append(rmse)
        inference_times.append(elapsed_ms)

    rmspe_values = np.asarray(rmspe_values, dtype=float)
    inference_times = np.asarray(inference_times, dtype=float)
    print()
    print(f"Mean RMSPE : {float(np.mean(rmspe_values)):.3f} deg")
    print(f"Max RMSPE  : {float(np.max(rmspe_values)):.3f} deg")
    print(f"Min RMSPE  : {float(np.min(rmspe_values)):.3f} deg")
    print(f"Mean infer : {float(np.mean(inference_times)):.2f} ms/sample")


def resolve_batch_dataset_path(args: argparse.Namespace) -> Path:
    if args.batch_dataset_path is not None:
        return args.batch_dataset_path
    if args.dataset_path != DEFAULT_DATASET_PATH:
        return args.dataset_path
    if DEFAULT_BATCH_DATASET_PATH.exists():
        return DEFAULT_BATCH_DATASET_PATH
    return DEFAULT_DATASET_PATH


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model_path = resolve_model_path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Radar model file not found: {model_path}. Run python main_radar.py first."
        )
    dataset_path = resolve_batch_dataset_path(args) if args.batch is not None else args.dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Radar dataset file not found: {dataset_path}. Run python main_radar.py first."
        )

    dataset = torch_load(dataset_path, torch.device("cpu"))
    X, truth_rad = dataset[0]
    num_sources = int(truth_rad.numel())

    model, tau = load_model(model_path, device, num_sources)
    data_handler.device = device

    print("Radar SubspaceNet DoA Demo")
    print("==========================")
    print(f"Model       : {model_path}")
    print(f"Dataset     : {dataset_path}")
    print(f"Device      : {device}")

    if args.batch is not None:
        run_batch(dataset, model, tau, device, args.batch)
        return

    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(
            f"sample-index must be in [0, {len(dataset) - 1}], got {args.sample_index}."
        )

    X, truth_rad = dataset[args.sample_index]
    pred_deg, _ = predict_doa(model, X, tau, device)
    truth_deg = np.rad2deg(truth_rad.detach().cpu().numpy())
    matched_pred, signed_error, rmse = best_periodic_match(pred_deg, truth_deg)
    abs_error = np.abs(signed_error)

    print(f"Sample index: {args.sample_index}")
    print(f"Signal X    : shape={tuple(X.shape)}, dtype={X.dtype}")
    print()
    print("DoA angles are shown in degrees.")
    print(f"Ground truth: {format_values(truth_deg)}")
    print(f"Prediction : {format_values(matched_pred)}")
    print(f"Abs. error : {format_values(abs_error)}")
    print(f"RMSE       : {rmse:.3f} deg")
    print(f"MAE        : {float(np.mean(abs_error)):.3f} deg")


if __name__ == "__main__":
    main()
