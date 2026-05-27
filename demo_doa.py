"""Minimal SubspaceNet DoA demo.

Loads a trained SubspaceNet model, reads one array observation from the bundled
test dataset, and prints the DoA estimate against ground truth.
"""

import argparse
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

import src.data_handler as data_handler
from src.models import SubspaceNet


DEFAULT_MODEL_PATH = (
    Path("data")
    / "weights"
    / "final_models"
    / "SubspaceNet_M=5_T=100_SNR_10_tau=8_NarrowBand_diff_method=esprit_coherent_eta=0_sv_noise=0"
)
DEFAULT_DATASET_PATH = (
    Path("data")
    / "datasets"
    / "diff_esprit"
    / "test"
    / "Generic_DataSet_NarrowBand_coherent_100_M=5_N=8_T=100_SNR=10_eta=0_sv_noise_var0_.h5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal SubspaceNet DoA estimation demo."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained SubspaceNet state_dict.",
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
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device.",
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")

    dataset = torch_load(args.dataset_path, torch.device("cpu"))
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(
            f"sample-index must be in [0, {len(dataset) - 1}], got {args.sample_index}."
        )

    X, truth_rad = dataset[args.sample_index]
    if not torch.is_complex(X):
        raise ValueError("Expected the dataset sample X to be a complex array signal.")

    state_dict = torch_load(args.model_path, device)
    tau = infer_tau(state_dict)
    num_sources = int(truth_rad.numel())

    model = SubspaceNet(tau=tau, M=num_sources, diff_method="esprit").to(device)
    model.load_state_dict(state_dict)
    model.eval()

    data_handler.device = device
    with torch.no_grad():
        model_input = data_handler.create_autocorrelation_tensor(X, tau)
        model_input = model_input.unsqueeze(0).to(device=device, dtype=torch.float32)
        pred_rad, _, _, _ = model(model_input)

    pred_deg = np.rad2deg(pred_rad.squeeze(0).detach().cpu().numpy())
    truth_deg = np.rad2deg(truth_rad.detach().cpu().numpy())
    matched_pred, signed_error, rmse = best_periodic_match(pred_deg, truth_deg)
    abs_error = np.abs(signed_error)

    print("SubspaceNet DoA Demo")
    print("====================")
    print(f"Model       : {args.model_path}")
    print(f"Dataset     : {args.dataset_path}")
    print(f"Sample index: {args.sample_index}")
    print(f"Signal X    : shape={tuple(X.shape)}, dtype={X.dtype}")
    print(f"Device      : {device}")
    print()
    print("DoA angles are shown in degrees.")
    print(f"Ground truth: {format_values(truth_deg)}")
    print(f"Prediction : {format_values(matched_pred)}")
    print(f"Abs. error : {format_values(abs_error)}")
    print(f"RMSE       : {rmse:.3f} deg")
    print(f"MAE        : {float(np.mean(abs_error)):.3f} deg")


if __name__ == "__main__":
    main()
