import logging
import os
import numpy as np
import torch
from functools import partial
import multiprocessing as mp

from giems_lstm.config import Config
from giems_lstm.data import (
    extract_window_data,
    wetland_dataloader,
    WetlandDataset,
    fit_scalers,
    transform_target,
    transform_features,
)
from giems_lstm.engine import Trainer, Evaluator
from .allocate_coords import _allocate_coords


def _train_task(
    coord: tuple[int, int],
    config: Config,
    thread_id: int,
    device: torch.device,
    left: bool = False,
):
    logger = logging.getLogger()

    # --- Training Logic ---
    lat_idx, lon_idx = coord
    model_path = config.model_folder / f"{lat_idx}_{lon_idx}.pth"
    if not config.sys.debug and model_path.exists() and not config.sys.cover_exist:
        logger.info(f"[Thread {thread_id}] Model {lat_idx},{lon_idx} exists. Skipping.")
        return
    if left and not model_path.exists():
        # Create empty .pth file to reserve the spot
        torch.save({}, model_path)
        logger.info(f"[Thread {thread_id}] Created placeholder for {lat_idx},{lon_idx}")

    logger.info(f"[Thread {thread_id}] Training {lat_idx}, {lon_idx}")

    # Construct Dataset
    target, features, dates = extract_window_data(
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
    target_scaled = transform_target(target, target_scaler)
    features_scaled = transform_features(features, feature_scalers)

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
            eval_folder=config.eval_folder,
            target_scaler=target_scaler,
            device=device,
            debug=config.sys.debug,
        )
        evaluator.run()
    except Exception as e:
        logger.error(f"Error during evaluation for {lat_idx},{lon_idx}: {e}")
        return


def _train(thread_id: int, config_path: str, debug: bool, para: int, left: bool):
    if left:
        thread_id = -1  # Dummy value to indicate left mode

    # Each process needs to reload configuration and data references to avoid issues with multiprocessing sharing complex objects forked from the main process
    config = Config(config_path=config_path, mode="train")
    if debug:
        config.sys.debug = True
        config.train.n_epochs = 10  # Reduce epochs in debug mode
    # Determine the range of tasks this process is responsible for
    mask = config.mask
    total_tasks = int(np.sum(mask))
    tasks_per_thread = config.sys.tasks_per_thread

    start_task = thread_id * tasks_per_thread if not left else 0
    end_task = (
        min((thread_id + 1) * tasks_per_thread, total_tasks)
        if not left
        else total_tasks
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_t = partial(_train_task, config=config, thread_id=thread_id, device=device)

    train_coords = _allocate_coords(mask, start_task, end_task, left=left)
    os.makedirs(config.model_folder, exist_ok=True)
    if para <= 1 and not left:
        for i, (lat_idx, lon_idx) in enumerate(train_coords):
            logging.info(f"Processing task {i + start_task + 1}/{end_task}")
            train_t((lat_idx, lon_idx))
    elif para > 1 and not left:
        with mp.Pool(processes=para) as pool:
            pool.map(train_t, train_coords)
    elif left:
        for i, (lat_idx, lon_idx) in enumerate(train_coords):
            logging.info(f"Processing task {lat_idx},{lon_idx} in left mode")
            train_t((lat_idx, lon_idx), left=True)
