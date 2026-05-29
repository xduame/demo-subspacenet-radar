"""Radar DoA demo.

Modes:
  preset      Run three fixed DoA scenes.
  batch       Evaluate the saved 500-sample radar test set.
  interactive Estimate user-provided DoA angles.
  file        Estimate DoA from a saved signal file.
"""

import argparse
from itertools import permutations
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm

import src.data_handler as data_handler
from src.data_handler import create_autocorrelation_tensor, read_data
from src.models import SubspaceNet
from src.signal_creation import Samples
from src.system_model import SystemModelParams


TAU = 8
NUM_SOURCES = 2
MODEL_PATH = (
    Path("data")
    / "weights"
    / "final_models"
    / "Radar_SubspaceNet_M=2_T=200_SNR_0_tau=8_NarrowBand_diff_method=root_music_non-coherent_eta=0_bias=0_sv_noise=0.pt"
)
TEST_DATASET_PATH = (
    Path("data")
    / "datasets"
    / "radar_baseline"
    / "test"
    / "SubspaceNet_DataSet_NarrowBand_non-coherent_500_M=2_N=16_T=200_SNR=0_eta=0_sv_noise_var0_bias=0_.h5"
)
PRESET_DOAS = [
    [-40.0, 20.0],
    [-20.0, 35.0],
    [-45.0, 50.0],
]


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_system_model_params() -> SystemModelParams:
    return (
        SystemModelParams()
        .set_parameter("N", 16)
        .set_parameter("M", NUM_SOURCES)
        .set_parameter("T", 200)
        .set_parameter("snr", 0)
        .set_parameter("signal_type", "NarrowBand")
        .set_parameter("signal_nature", "non-coherent")
        .set_parameter("eta", 0)
        .set_parameter("bias", 0)
        .set_parameter("sv_noise_var", 0)
        .set_parameter("data_source", "radar")
        .set_parameter("lib_path", "data/data_manual.xlsx")
        .set_parameter("fs_mhz", 200)
        .set_parameter("rf_center_mhz", 9000)
        .set_parameter("modes_pool", ["VS", "MRWS", "TASS", "TAST"])
    )


def load_model(device: torch.device) -> SubspaceNet:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model weights: {MODEL_PATH}")
    model = SubspaceNet(tau=TAU, M=NUM_SOURCES, diff_method="root_music").to(device)
    model.load_state_dict(torch_load(MODEL_PATH, device))
    model.eval()
    return model


def best_match_deg(pred_deg: np.ndarray, truth_deg: np.ndarray):
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


def make_radar_signal(doa_deg: list[float]) -> tuple[torch.Tensor, np.ndarray]:
    params = build_system_model_params()
    samples_model = Samples(params)
    if params.data_source == "radar":
        from src.radar_samples import RadarSamples

        samples_model = RadarSamples(params)
    samples_model.set_doa(doa_deg)
    X = torch.tensor(
        samples_model.samples_creation(
            noise_mean=0,
            noise_variance=1,
            signal_mean=0,
            signal_variance=1,
        )[0],
        dtype=torch.complex64,
    )
    return X, np.asarray(doa_deg, dtype=float)


def _to_signal_tensor(array: np.ndarray) -> torch.Tensor:
    array = np.asarray(array)
    if array.shape != (16, 200):
        raise ValueError(f"Expected X shape (16, 200), got {array.shape}.")
    if not np.iscomplexobj(array):
        raise ValueError("Expected X to contain complex I/Q samples.")
    return torch.tensor(array, dtype=torch.complex64)


def _read_truth_deg(npz_data) -> np.ndarray | None:
    if "doa_deg" in npz_data:
        truth_deg = np.asarray(npz_data["doa_deg"], dtype=float)
    elif "truth_deg" in npz_data:
        truth_deg = np.asarray(npz_data["truth_deg"], dtype=float)
    elif "doa_rad" in npz_data:
        truth_deg = np.rad2deg(np.asarray(npz_data["doa_rad"], dtype=float))
    else:
        return None
    if truth_deg.shape != (NUM_SOURCES,):
        raise ValueError(f"Expected truth DoA shape ({NUM_SOURCES},), got {truth_deg.shape}.")
    return truth_deg


def load_signal_file(input_path: Path) -> tuple[torch.Tensor, np.ndarray | None]:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing signal file: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".npz":
        with np.load(input_path) as data:
            if "X" not in data:
                raise ValueError("NPZ signal file must contain an 'X' array.")
            X = _to_signal_tensor(data["X"])
            truth_deg = _read_truth_deg(data)
        return X, truth_deg
    if suffix == ".npy":
        return _to_signal_tensor(np.load(input_path)), None

    raise ValueError("Unsupported signal file type. Use .npz or .npy.")


def predict_from_signal(model: SubspaceNet, X: torch.Tensor, device: torch.device):
    data_handler.device = device
    model_input = create_autocorrelation_tensor(X, TAU)
    model_input = model_input.unsqueeze(0).to(device=device, dtype=torch.float32)
    start = perf_counter()
    with torch.no_grad():
        pred_rad, _, _, _ = model(model_input)
    elapsed_ms = (perf_counter() - start) * 1000.0
    pred_deg = np.rad2deg(pred_rad.squeeze(0).detach().cpu().numpy())
    return pred_deg, elapsed_ms


def print_case(title: str, pred_deg: np.ndarray, truth_deg: np.ndarray, elapsed_ms: float):
    matched_pred, signed_error, rmse = best_match_deg(pred_deg, truth_deg)
    abs_error = np.abs(signed_error)
    print(f"\n{title}")
    print(f"Ground truth: {format_values(truth_deg)} deg")
    print(f"Prediction  : {format_values(matched_pred)} deg")
    print(f"Abs. error  : {format_values(abs_error)} deg")
    print(f"RMSPE/RMSE  : {rmse:.3f} deg")
    print(f"Inference   : {elapsed_ms:.2f} ms")


def print_prediction(title: str, pred_deg: np.ndarray, elapsed_ms: float) -> None:
    print(f"\n{title}")
    print(f"Prediction  : {format_values(np.sort(pred_deg))} deg")
    print(f"Inference   : {elapsed_ms:.2f} ms")


def run_preset(model: SubspaceNet, device: torch.device) -> None:
    print("Preset scenes")
    for index, doa in enumerate(PRESET_DOAS, start=1):
        X, truth_deg = make_radar_signal(doa)
        pred_deg, elapsed_ms = predict_from_signal(model, X, device)
        print_case(f"Scene {index}", pred_deg, truth_deg, elapsed_ms)


def run_batch(model: SubspaceNet, device: torch.device, limit: int = 500) -> None:
    if not TEST_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing test dataset: {TEST_DATASET_PATH}. Run python main_radar.py first."
        )
    dataset = read_data(TEST_DATASET_PATH)
    count = min(limit, len(dataset))
    rmspe_values = []
    inference_times = []
    print(f"Batch evaluation: {count} samples")
    for model_input, truth_rad in tqdm(dataset[:count], desc="Evaluating"):
        model_input = model_input.unsqueeze(0).to(device=device, dtype=torch.float32)
        start = perf_counter()
        with torch.no_grad():
            pred_rad, _, _, _ = model(model_input)
        inference_times.append((perf_counter() - start) * 1000.0)
        pred_deg = np.rad2deg(pred_rad.squeeze(0).detach().cpu().numpy())
        truth_deg = np.rad2deg(truth_rad.detach().cpu().numpy())
        _, _, rmse = best_match_deg(pred_deg, truth_deg)
        rmspe_values.append(rmse)

    mean_rmspe_deg = float(np.mean(rmspe_values))
    mean_inference_ms = float(np.mean(inference_times))
    print(f"Test RMSPE : {mean_rmspe_deg:.3f} deg")
    print(f"Mean infer : {mean_inference_ms:.2f} ms/sample")


def run_file(model: SubspaceNet, device: torch.device, input_path: Path | None) -> None:
    if input_path is None:
        raise SystemExit("--input-path is required for --mode file.")
    X, truth_deg = load_signal_file(input_path)
    pred_deg, elapsed_ms = predict_from_signal(model, X, device)
    if truth_deg is None:
        print_prediction("File scene", pred_deg, elapsed_ms)
    else:
        print_case("File scene", pred_deg, truth_deg, elapsed_ms)


def parse_angles(raw_values: str | list[str]) -> list[float]:
    if isinstance(raw_values, str):
        raw_values = raw_values.replace(",", " ").split()
    elif len(raw_values) == 1:
        raw_values = raw_values[0].replace(",", " ").split()
    if len(raw_values) != NUM_SOURCES:
        raise ValueError(f"Please provide exactly {NUM_SOURCES} angles.")
    angles = [float(value) for value in raw_values]
    for angle in angles:
        if angle < -60.0 or angle > 60.0:
            raise ValueError("Angles must be in [-60, 60] degrees.")
    if len(set(round(angle, 6) for angle in angles)) != len(angles):
        raise ValueError("Angles must be distinct.")
    return angles


def run_interactive(
    model: SubspaceNet, device: torch.device, angle_args: str | None = None
) -> None:
    while True:
        try:
            if angle_args:
                angles = parse_angles(angle_args)
            else:
                raw = input("Enter two DoA angles in [-60, 60] degrees: ")
                angles = parse_angles(raw)
            break
        except ValueError as exc:
            print(f"Invalid input: {exc}")
            if angle_args:
                raise SystemExit(2)
    X, truth_deg = make_radar_signal(angles)
    pred_deg, elapsed_ms = predict_from_signal(model, X, device)
    print_case("Interactive scene", pred_deg, truth_deg, elapsed_ms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Radar SubspaceNet DoA demo.")
    parser.add_argument(
        "--mode",
        choices=("preset", "batch", "interactive", "file", "all"),
        default="preset",
        help="Demo mode to run.",
    )
    parser.add_argument(
        "--batch-limit",
        type=int,
        default=500,
        help="Number of saved test samples for batch mode.",
    )
    parser.add_argument(
        "--angles",
        help='Two angles for interactive mode, e.g. --angles "-20 35".',
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        help="Signal file for file mode. Supports .npz with X/doa_deg or .npy with X.",
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
        raise RuntimeError("CUDA was requested, but it is not available.")
    return torch.device(device_name)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model = load_model(device)
    print(f"Model : {MODEL_PATH}")
    print(f"Device: {device}")

    if args.mode in ("preset", "all"):
        run_preset(model, device)
    if args.mode in ("batch", "all"):
        run_batch(model, device, limit=args.batch_limit)
    if args.mode == "interactive":
        run_interactive(model, device, angle_args=args.angles)
    if args.mode == "file":
        run_file(model, device, input_path=args.input_path)
    elif args.mode == "all" and args.angles:
        run_interactive(model, device, angle_args=args.angles)


if __name__ == "__main__":
    main()
