from __future__ import annotations

import typer
import os
import sys
import random
import multiprocessing as mp
from functools import partial
import logging

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import numpy as np
    import pandas as pd
    import xarray as xr
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    from giems_lstm.data import (
        extract_window_data,
        fit_scalers,
        transform_features,
        transform_target,
        WetlandDataset,
        wetland_dataloader,
    )
    from giems_lstm.engine import Trainer, Evaluator, Predictor
    from giems_lstm.config import Config
    from giems_lstm.model import LSTMNetKAN

app = typer.Typer(help="GIEMS-LSTM Training and Prediction CLI")


def _init_imports():
    pass


def setup_global_logging(debug: bool, process_type: str):
    """
    Configures the root logger for the current process.
    - debug: If True, sets level to DEBUG. Otherwise, sets to INFO.
    - process_type: A label (e.g., 'Main' or 'Worker-ID') for log differentiation.
    """
    # set logging level based on debug flag
    level = logging.DEBUG if debug else logging.INFO

    # fetch the root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Avoid adding multiple handlers in multiprocessing
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Define format including process info for distinguishing logs in parallel runs
        formatter = logging.Formatter(
            f"[%(levelname)s] [PROC:{process_type}] [%(filename)s:%(lineno)d] - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def _seed_everything(seed: int):
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


def _train_task(
    coord: tuple[int, int], config: Config, thread_id: int, device: torch.device
):
    logger = logging.getLogger()

    # --- Training Logic ---
    lat_idx, lon_idx = coord
    model_path = config.model_folder / f"{lat_idx}_{lon_idx}.pth"
    if not config.sys.debug and model_path.exists() and not config.sys.cover_exist:
        logger.info(f"[Thread {thread_id}] Model {lat_idx},{lon_idx} exists. Skipping.")
        return

    logger.info(f"[Thread {thread_id}] Training {lat_idx}, {lon_idx}")

    # Construct Dataset
    features, target, dates = extract_window_data(
        lat_idx,
        lon_idx,
        config.TVARs,
        config.CVARs,
        config.model.window_size,
        config.train.start_date,
        config.train.end_date,
    )
    target_scaler, feature_scalers = fit_scalers(
        raw_target=target, raw_features=features
    )
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
            train_config=config.train,
            model_config=config.model,
            lat_idx=lat_idx,
            lon_idx=lon_idx,
            model_folder=config.model_folder,
            device=device,
            debug=config.sys.debug,
        )
        trainer.run()

    except Exception as e:
        logger.error(f"Error preparing data for {lat_idx},{lon_idx}: {e}")
        return

    try:
        evaluator = Evaluator(
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            lat_idx=lat_idx,
            lon_idx=lon_idx,
            model=trainer.model,
            device=device,
            debug=config.sys.debug,
        )
        evaluator.run()
    except Exception as e:
        logger.error(f"Error during evaluation for {lat_idx},{lon_idx}: {e}")
        return


def _allocate_coords(mask: np.ndarray, start_task: int, end_task: int):
    task_counter = 0
    for lat_idx in range(mask.shape[0]):
        for lon_idx in range(mask.shape[1]):
            if not mask[lat_idx, lon_idx]:
                continue
            if task_counter < start_task:
                task_counter += 1
                continue
            elif task_counter >= end_task:
                return

            task_counter += 1
            yield (lat_idx, lon_idx)


def _train(thread_id: int, config_path: str, debug: bool, para: int):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_t = partial(_train_task, config=config, thread_id=thread_id, device=device)

    train_coords = _allocate_coords(mask, start_task, end_task)
    os.makedirs(config.model_folder, exist_ok=True)
    if para <= 1:
        for lat_idx, lon_idx in train_coords:
            train_t((lat_idx, lon_idx))
    else:
        with mp.Pool(processes=para) as pool:
            pool.map(train_t, train_coords)


def _predict_task(
    coord: tuple[int, int], config: Config, thread_id: int, device: torch.device
):
    logger = logging.getLogger()

    lat_idx, lon_idx = coord
    logger.info(f"[Thread {thread_id}] Predicting {lat_idx}, {lon_idx}")

    train_features, train_target, _ = extract_window_data(
        lat_idx,
        lon_idx,
        config.TVARs,
        config.CVARs,
        config.model.window_size,
        config.predict.train_start_date,
        config.predict.train_end_date,
    )
    feature_scalers, target_scaler = fit_scalers(train_features, train_target)
    pred_features, _, pred_dates = extract_window_data(
        lat_idx,
        lon_idx,
        config.TVARs,
        config.CVARs,
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
    if not config.sys.debug and save_path.exists() and not config.sys.cover_exist:
        logger.info(
            f"Predictions for location ({lat_idx}, {lon_idx}) already exist. Skipping..."
        )
        return
    try:
        model = LSTMNetKAN(
            input_dim=(len(config.TVARs) + len(config.CVARs) - 1)
            * (config.model.window_size**2)
            + 1,
            hidden_dim=config.model.hidden_dim,
            output_dim=1,
            n_layers=config.model.n_layers,
            device=device,
        )
        model.load_state_dict(
            torch.load(
                config.model_folder / f"{lat_idx}_{lon_idx}.pth", map_location=device
            )
        )
    except Exception as e:
        logger.error(f"Error loading model for {lat_idx},{lon_idx}: {e}")
        return

    try:
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
    except Exception as e:
        logger.error(f"Error during prediction for {lat_idx},{lon_idx}: {e}")
        return


def _predict(thread_id: int, config_path: str, debug: bool, para: int):
    # Each process needs to reload configuration and data references to avoid issues with multiprocessing sharing complex objects forked from the main process
    config = Config(config_path=config_path, mode="predict")
    if debug:
        config.sys.debug = True
    # Determine the range of tasks this process is responsible for
    mask = config.mask
    total_tasks = int(np.sum(mask))
    tasks_per_thread = config.sys.tasks_per_thread

    start_task = thread_id * tasks_per_thread
    end_task = min((thread_id + 1) * tasks_per_thread, total_tasks)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict_t = partial(
        _predict_task, config=config, thread_id=thread_id, device=device
    )

    predict_coords = _allocate_coords(mask, start_task, end_task)
    os.makedirs(config.pred_folder, exist_ok=True)
    if para <= 1:
        for lat_idx, lon_idx in predict_coords:
            predict_t((lat_idx, lon_idx))
    else:
        with mp.Pool(processes=para) as pool:
            pool.map(predict_t, predict_coords)


def _parse_filename(filename: Path):
    """
    Extract lat_idx, lon_idx from filename 'lat_lon.npy'.
    Returns (lat_idx, lon_idx) or None if format is invalid.
    """
    try:
        stem = filename.stem
        parts = stem.split("_")
        if len(parts) != 2:
            return None
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _collect_batch(file_batch: list[str], zero_array: np.ndarray):
    count = 0
    for file_path in file_batch:
        indices = _parse_filename(file_path)
        if not indices:
            continue

        lat_idx, lon_idx = indices

        try:
            # Load data
            data = np.load(file_path)
            zero_array[:, lat_idx, lon_idx] = data
            count += 1

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return count


def _collect(config_path: str, parallel: int):
    logger = logging.getLogger()

    config = Config(config_path=config_path, mode="analyze")
    pred_folder = config.pred_folder
    output_file = config.predict.output_file

    if not pred_folder.exists():
        logger.error(f"Prediction folder {pred_folder} does not exist.")
        sys.exit(1)

    logger.debug(f"Scanning files in {pred_folder}...")
    files = list(pred_folder.glob("*.npy"))
    if not files:
        logger.error("No .npy files found.")
        sys.exit(0)

    GIEMS = config.TVARs["giems2"]
    lats = GIEMS.coords["lat"].values
    lons = GIEMS.coords["lon"].values
    dates = pd.date_range(
        start=config.predict.start_date,
        end=config.predict.end_date,
        freq="MS",
    )

    zero_array = np.full((len(dates), len(lats), len(lons)), np.nan, dtype=np.float32)
    coords = _allocate_coords(config.mask, 0, config.total_tasks)

    if parallel <= 1:
        for lat, lon in coords:
            file_path = pred_folder / f"{lat}_{lon}.npy"
            if not file_path.exists():
                logger.warning(f"File {file_path} does not exist. Skipping.")
                continue
            data = np.load(file_path)
            zero_array[:, lat, lon] = data.item()["prediction"]
    else:
        chunk_size = config.sys.tasks_per_thread
        batches = [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]
        total_loaded = 0

        with ThreadPoolExecutor(max_workers=parallel) as executor:
            future_to_batch = {
                executor.submit(_collect_batch, batch, zero_array): batch
                for batch in batches
            }

            for future in as_completed(future_to_batch):
                try:
                    count = future.result()
                    total_loaded += count
                    logger.debug(f"Loaded {total_loaded}/{config.total_tasks} files...")
                except Exception as e:
                    logger.error(f"Error loading batch: {e}")

    logger.info(f"All files loaded. Saving to {output_file}...")
    ds = xr.Dataset(
        data_vars={"fwet": (("time", "lat", "lon"), zero_array)},
        coords={"time": dates, "lat": lats, "lon": lons},
        attrs={"description": "GIEMS-LSTM Predicted Wetland Fraction"},
    )
    encoding = {"fwet": {"zlib": True, "complevel": 5, "_FillValue": np.nan}}
    ds.to_netcdf(output_file, encoding=encoding)

    logger.info("Collection and saving complete.")


# ==============================================================================
# CLI Commands
# ==============================================================================
def _uniform_entry(debug: bool, parallel: int, seed: int):
    _init_imports()

    setup_global_logging(debug, "Main")
    logger = logging.getLogger()
    if debug:
        logger.warning("!!! DEBUG MODE ENABLED !!!")

    if parallel >= 1:
        num_workers = mp.cpu_count() // parallel
        if num_workers < 1:
            logger.warning(
                f"Requested {parallel} processes exceeds CPU count. Setting to max available."
            )
            num_workers = 1
        logger.warning(
            f"Spawning {parallel} local processes with {num_workers} workers each."
        )
        torch.set_num_threads(num_workers)

    # Set seed for the main process
    _seed_everything(seed)


@app.command()
def train(
    config_path: str = typer.Option(
        "config/F.toml", "--config", "-c", help="Path to config file"
    ),
    thread_id: int = typer.Option(
        0,
        "--thread-id",
        "-t",
        help="Specific chunk ID to run (for manual/slurm splitting)",
    ),
    parallel: int = typer.Option(
        0,
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
    Train GIEMS-LSTM models for all locations specified in the config mask.
    """

    _uniform_entry(debug, parallel, seed)

    _train(thread_id, config_path, debug, parallel)


@app.command()
def predict(
    config_path: str = typer.Option(
        "config/F.toml", "--config", "-c", help="Path to config file"
    ),
    thread_id: int = typer.Option(
        0, "--thread-id", "-t", help="Specific chunk ID to run"
    ),
    parallel: int = typer.Option(
        0, "--parallel", "-p", help="Number of local processes to spawn."
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
    seed: int = typer.Option(
        3407, "--seed", "-s", help="Random seed for reproducibility"
    ),
):
    """
    Predict GIEMS-LSTM models for all locations specified in the config mask.
    """
    _uniform_entry(debug, parallel, seed)

    _predict(thread_id, config_path, debug, parallel)


@app.command()
def collect(
    config_path: str = typer.Option(
        "config/F.toml", "--config", "-c", help="Path to config file"
    ),
    parallel: int = typer.Option(
        4, "--parallel", "-p", help="Number of threads to use for collection."
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
    seed: int = typer.Option(
        3407, "--seed", "-s", help="Random seed for reproducibility"
    ),
):
    """
    Collect predicted results into a single NetCDF file.
    """
    _uniform_entry(debug, parallel, seed)

    _collect(config_path, parallel)


if __name__ == "__main__":
    app()
