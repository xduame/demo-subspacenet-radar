"""Radar SubspaceNet training entrypoint.

This script keeps the original SubspaceNet training flow and only changes the
system/data configuration to use RadarSamples as the signal source.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import torch

from src.criterions import set_criterions
from src.data_handler import create_dataset
from src.evaluation import evaluate
from src.models import ModelGenerator
from src.plotting import initialize_figures
from src.system_model import SystemModelParams
from src.training import TrainingParams, get_simulation_filename, simulation_summary, train
from src.utils import set_unified_seed


SAMPLES_SIZE = 5000
TRAIN_TEST_RATIO = 0.1

EPOCHS = 15
BATCH_SIZE = 128
LR = 1e-4
WEIGHT_DECAY = 1e-9
TAU = 8
DIFF_METHOD = "esprit"


def build_system_model_params() -> SystemModelParams:
    return (
        SystemModelParams()
        .set_parameter("N", 16)
        .set_parameter("M", 2)
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


def main() -> None:
    warnings.simplefilter("ignore")
    plt.close("all")
    set_unified_seed()

    external_data_path = Path.cwd() / "data"
    datasets_path = external_data_path / "datasets" / "radar_baseline"
    simulations_path = external_data_path / "simulations"
    final_models_path = external_data_path / "weights" / "final_models"

    (datasets_path / "train").mkdir(parents=True, exist_ok=True)
    (datasets_path / "test").mkdir(parents=True, exist_ok=True)
    simulations_path.mkdir(parents=True, exist_ok=True)
    final_models_path.mkdir(parents=True, exist_ok=True)

    system_model_params = build_system_model_params()
    model_config = (
        ModelGenerator()
        .set_model_type("SubspaceNet")
        .set_diff_method(DIFF_METHOD)
        .set_tau(TAU)
        .set_model(system_model_params)
    )
    simulation_filename = (
        "Radar_" + get_simulation_filename(system_model_params, model_config) + ".pt"
    )

    print("------------------------------------")
    print("------ Radar SubspaceNet Train ------")
    print("------------------------------------")
    print(f"Dataset path: {datasets_path}")
    print(f"Model path  : {final_models_path / simulation_filename}")

    print("\nCreating radar datasets...")
    train_dataset, _, _ = create_dataset(
        system_model_params=system_model_params,
        samples_size=SAMPLES_SIZE,
        model_type=model_config.model_type,
        tau=model_config.tau,
        save_datasets=True,
        datasets_path=datasets_path,
        true_doa=None,
        phase="train",
    )
    test_dataset, generic_test_dataset, samples_model = create_dataset(
        system_model_params=system_model_params,
        samples_size=int(TRAIN_TEST_RATIO * SAMPLES_SIZE),
        model_type=model_config.model_type,
        tau=model_config.tau,
        save_datasets=True,
        datasets_path=datasets_path,
        true_doa=None,
        phase="test",
    )

    training_parameters = (
        TrainingParams()
        .set_batch_size(BATCH_SIZE)
        .set_epochs(EPOCHS)
        .set_model(model=model_config)
        .set_optimizer(optimizer="Adam", learning_rate=LR, weight_decay=WEIGHT_DECAY)
        .set_training_dataset(train_dataset)
        .set_schedular(step_size=EPOCHS, gamma=0.2)
        .set_criterion()
    )
    simulation_summary(
        system_model_params=system_model_params,
        model_type=model_config.model_type,
        parameters=training_parameters,
        phase="training",
    )
    model, _, _ = train(
        training_parameters=training_parameters,
        model_name=simulation_filename,
        plot_curves=False,
        saving_path=final_models_path,
    )

    criterion, subspace_criterion = set_criterions("rmse")
    model_test_dataset = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, drop_last=False
    )
    generic_test_dataset = torch.utils.data.DataLoader(
        generic_test_dataset, batch_size=1, shuffle=False, drop_last=False
    )

    simulation_summary(
        system_model_params=system_model_params,
        model_type=model_config.model_type,
        phase="evaluation",
        parameters=training_parameters,
    )
    evaluate(
        model=model,
        model_type=model_config.model_type,
        model_test_dataset=model_test_dataset,
        generic_test_dataset=generic_test_dataset,
        criterion=criterion,
        subspace_criterion=subspace_criterion,
        system_model=samples_model,
        figures=initialize_figures(),
        plot_spec=False,
    )
    print(f"\nSaved best model: {final_models_path / simulation_filename}")


if __name__ == "__main__":
    main()
