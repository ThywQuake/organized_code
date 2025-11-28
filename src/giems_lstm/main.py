import typer
import os
import sys
import torch
import numpy as np
import random
import multiprocessing as mp
from pathlib import Path
from functools import partial

# Make sure the project root is in sys.path for imports
# Assuming this script is located at src/giems_lstm/main.py
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import project modules
from giems_lstm.config import Config
from giems_lstm.data import (
    WetlandDataset,
    wetland_dataloader,
    fit_scalers,
    transform_features,
    transform_target,
    extract_window_data,
)
from giems_lstm.engine import Trainer, Evaluator, Predictor
from giems_lstm.model import LSTMNetKAN


app = typer.Typer(help="GIEMS-LSTM Training and Prediction CLI")


def seed_everything(seed: int):
    """
    Set random seeds for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# Worker Functions (Must be top-level for multiprocessing pickle)
# ==============================================================================


def train_chunk(thread_id: int, config_path: str, debug: bool, seed: int):
    """
    Process a single chunk of training tasks (core logic from training.py)
    """
    # Set seed for this worker process
    seed_everything(seed)

    # Each process needs to reload configuration and data references to avoid issues with multiprocessing sharing complex objects forked from the main process
    config = Config(config_path=config_path, mode="train")
    if debug:
        config.sys.debug = True
        config.train.n_epochs = 10  # Reduce epochs in debug mode
    # Determine the range of tasks this process is responsible for
    mask = config.mask
    total_tasks = int(np.sum(mask))
    tasks_per_thread = config.sys.tasks_per_thread

    start_task = thread_id * tasks_per_thread
    end_task = min((thread_id + 1) * tasks_per_thread, total_tasks)

    if start_task >= total_tasks:
        print(
            f"[Thread {thread_id}] Start task {start_task} exceeds total {total_tasks}. Exiting."
        )
        return

    print(f"--- [Thread {thread_id}] Processing tasks {start_task} to {end_task} ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Iterate over tasks
    task_counter = 0
    tasks_processed = 0

    # Preload data references (Lazy Loading)
    TVARs = config.TVARs
    CVARs = config.CVARs

    # Record all processed coordinates in the current chunk for subsequent Evaluation
    processed_coords = []

    for lat_idx in range(mask.shape[0]):
        for lon_idx in range(mask.shape[1]):
            if not mask[lat_idx, lon_idx]:
                continue

            # Skip tasks not belonging to the current chunk
            if task_counter < start_task:
                task_counter += 1
                continue
            elif task_counter >= end_task:
                break

            task_counter += 1
            processed_coords.append((lat_idx, lon_idx))

            # --- Training Logic ---
            os.makedirs(config.model_folder, exist_ok=True)
            model_path = config.model_folder / f"{lat_idx}_{lon_idx}.pth"

            if (
                not config.sys.debug
                and model_path.exists()
                and not config.sys.cover_exist
            ):
                print(
                    f"[Thread {thread_id}] Model {lat_idx},{lon_idx} exists. Skipping."
                )
                continue

            print(f"[Thread {thread_id}] Training {lat_idx}, {lon_idx}")

            # Construct Dataset
            features, target, dates = extract_window_data(
                lat_idx,
                lon_idx,
                TVARs,
                CVARs,
                config.model.window_size,
                config.train.start_date,
                config.train.end_date,
            )
            feature_scalers, target_scaler = fit_scalers(features, target)
            features_scaled = transform_features(features, feature_scalers)
            target_scaled = transform_target(target, target_scaler)

            dataset = WetlandDataset(
                features_scaled=features_scaled,
                dates=dates,
                target_scaled=target_scaled,
                seq_length=config.model.seq_length,
                predict_mode=False,
            )
            train_loader, test_loader = wetland_dataloader(
                dataset=dataset,
                train_years=config.train.train_years,
            )

            try:
                trainer = Trainer(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    learn_rate=config.train.lr,
                    hidden_dim=config.model.hidden_dim,
                    n_layers=config.model.n_layers,
                    n_epochs=config.train.n_epochs,
                    model_type=config.model.type,
                    verbose_epoch=config.train.verbose_epoch,
                    patience=config.train.patience,
                    debug=config.sys.debug,
                    device=device,
                )
                model = trainer.run()
                # Save the trained model
                torch.save(model.state_dict(), model_path)

            except Exception as e:
                print(f"Error preparing data for {lat_idx},{lon_idx}: {e}")
                continue

            tasks_processed += 1

    # --- Evaluation Logic (Chunk End) ---
    # Only run evaluation if at least one task was processed in this chunk
    if tasks_processed > 0:
        print(f"[Thread {thread_id}] Running evaluation for processed tasks...")
        # Construct a single Evaluator for all processed coordinates
        input_dim = (len(TVARs) + len(CVARs) - 1) * (config.model.window_size**2) + 1
        model = LSTMNetKAN(
            input_dim=input_dim,
            hidden_dim=config.model.hidden_dim,
            output_dim=1,
            n_layers=config.model.n_layers,
            device=device,
        )

        for lat, lon in processed_coords:
            evaluator = Evaluator(
                model=model,
                lat_idx=lat,
                lon_idx=lon,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                eval_folder=config.eval_folder,
                model_folder=config.model_folder,
                target_scaler=target_scaler,
                device=device,
                debug=config.sys.debug,
            )
            evaluator.run()


def predict_chunk(thread_id: int, config_path: str, debug: bool, seed: int):
    """
    Process a single chunk of prediction tasks (core logic from predicting.py)
    """
    # Set seed for this worker process
    seed_everything(seed)

    config = Config(config_path=config_path, mode="predict")
    if debug:
        config.sys.debug = True

    mask = config.mask
    total_tasks = config.dataset.mask["total"]
    tasks_per_thread = config.sys.tasks_per_thread

    start_task = thread_id * tasks_per_thread
    end_task = min((thread_id + 1) * tasks_per_thread, total_tasks)

    print(f"--- [Thread {thread_id}] Predicting tasks {start_task} to {end_task} ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TVARs = config.TVARs
    CVARs = config.CVARs

    task_counter = 0
    for lat_idx in range(mask.shape[0]):
        for lon_idx in range(mask.shape[1]):
            if not mask[lat_idx, lon_idx]:
                continue

            if task_counter < start_task:
                task_counter += 1
                continue
            elif task_counter >= end_task:
                break

            task_counter += 1

            print(f"[Thread {thread_id}] Predicting {lat_idx}, {lon_idx}")

            train_features, train_target, _ = extract_window_data(
                lat_idx,
                lon_idx,
                TVARs,
                CVARs,
                config.model.window_size,
                config.predict.train_start_date,
                config.predict.train_end_date,
            )
            feature_scalers, target_scaler = fit_scalers(train_features, train_target)
            pred_features, _, pred_dates = extract_window_data(
                lat_idx,
                lon_idx,
                TVARs,
                CVARs,
                config.model.window_size,
                config.predict.start_date,
                config.predict.end_date,
            )
            pred_features_scaled = transform_features(pred_features, feature_scalers)
            dataset = WetlandDataset(
                features_scaled=pred_features_scaled,
                dates=pred_dates,
                target_scaled=None,
                seq_length=config.model.seq_length,
                predict_mode=True,
            )

            save_path = config.pred_folder / f"{lat_idx}_{lon_idx}.npy"
            if not config.sys.debug and save_path.exists():
                print(
                    f"Predictions for location ({lat_idx}, {lon_idx}) already exist. Skipping..."
                )
                continue

            model = LSTMNetKAN(
                input_dim=(len(TVARs) + len(CVARs) - 1) * (config.model.window_size**2)
                + 1,
                hidden_dim=config.model.hidden_dim,
                output_dim=1,
                n_layers=config.model.n_layers,
                device=device,
            )
            model.load_state_dict(
                torch.load(
                    config.model_folder / f"{lat_idx}_{lon_idx}.pt", map_location=device
                )
            )

            predictor = Predictor(
                lat_idx=lat_idx,
                lon_idx=lon_idx,
                dataset=dataset,
                target_scaler=target_scaler,
                model=model,
                device=device,
                batch_size=config.model.batch_size,
                save_path=save_path,
                debug=config.sys.debug,
            )
            predictor.run()


# ==============================================================================
# CLI Commands
# ==============================================================================


@app.command()
def train(
    config_path: str = typer.Option(
        "config/E.toml", "--config", "-c", help="Path to config file"
    ),
    thread_id: int = typer.Option(
        0,
        "--thread-id",
        "-t",
        help="Specific chunk ID to run (for manual/slurm splitting)",
    ),
    parallel: int = typer.Option(
        1,
        "--parallel",
        "-p",
        help="Number of local processes to spawn. If >1, ignores thread-id and runs ALL chunks.",
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Enable debug mode (overrides config)"
    ),
    seed: int = typer.Option(
        3407, "--seed", "-s", help="Random seed for reproducibility"
    ),
):
    """
    Run training pipeline.
    """
    # Set seed for the main process
    seed_everything(seed)

    if debug:
        print("!!! DEBUG MODE ENABLED !!!")

    # 1. (Local Parallel)
    if parallel > 1:
        # Load config to determine total tasks and chunk size
        temp_cfg = Config(config_path, mode="train")
        total_tasks = int(np.sum(temp_cfg.mask))
        tasks_per_thread = temp_cfg.sys.tasks_per_thread

        num_chunks = (total_tasks + tasks_per_thread - 1) // tasks_per_thread
        print(
            f"Parallel Execution: Launching {parallel} workers for {num_chunks} total chunks."
        )

        # Use 'spawn' to avoid CUDA issues on some platforms
        # Note: 'spawn' is the recommended start method for PyTorch/CUDA
        mp.set_start_method("spawn", force=True)

        func = partial(train_chunk, config_path=config_path, debug=debug, seed=seed)

        with mp.Pool(processes=parallel) as pool:
            pool.map(func, range(num_chunks))

    # 2. Single Chunk / Slurm Mode
    else:
        print(f"Single Thread Execution: Running chunk {thread_id}")
        train_chunk(thread_id, config_path, debug, seed)


@app.command()
def predict(
    config_path: str = typer.Option(
        "config/E.toml", "--config", "-c", help="Path to config file"
    ),
    thread_id: int = typer.Option(
        0, "--thread-id", "-t", help="Specific chunk ID to run"
    ),
    parallel: int = typer.Option(
        1, "--parallel", "-p", help="Number of local processes to spawn."
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
    seed: int = typer.Option(
        3407, "--seed", "-s", help="Random seed for reproducibility"
    ),
):
    """
    Run prediction pipeline.
    """
    # Set seed for the main process
    seed_everything(seed)

    if parallel > 1:
        temp_cfg = Config(config_path, mode="predict")
        # Note: Ensure the method to get mask total is consistent
        total_tasks = temp_cfg.mask.sum()  # Or get from config
        tasks_per_thread = temp_cfg.sys.tasks_per_thread
        num_chunks = (total_tasks + tasks_per_thread - 1) // tasks_per_thread

        print(
            f"Parallel Execution: Launching {parallel} workers for {num_chunks} total chunks."
        )
        mp.set_start_method("spawn", force=True)
        func = partial(predict_chunk, config_path=config_path, debug=debug, seed=seed)

        with mp.Pool(processes=parallel) as pool:
            pool.map(func, range(num_chunks))
    else:
        predict_chunk(thread_id, config_path, debug, seed)


if __name__ == "__main__":
    app()
