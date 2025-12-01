from __future__ import annotations

import typer
import multiprocessing as mp
import logging

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from giems_lstm.utils import (
        _setup_global_logging,
        _seed_everything,
        _train,
        _predict,
        _collect,
    )


app = typer.Typer(help="GIEMS-LSTM Training and Prediction CLI")


def _init_imports():
    # ruff: noqa: F401
    globals()["torch"] = __import__("torch")
    globals()["np"] = __import__("numpy")
    globals()["pd"] = __import__("pandas")
    globals()["xr"] = __import__("xarray")

    globals()["_setup_global_logging"] = __import__(
        "giems_lstm.utils", fromlist=["_setup_global_logging"]
    )._setup_global_logging
    globals()["_seed_everything"] = __import__(
        "giems_lstm.utils", fromlist=["_seed_everything"]
    )._seed_everything
    globals()["_train"] = __import__("giems_lstm.utils", fromlist=["_train"])._train
    globals()["_predict"] = __import__(
        "giems_lstm.utils", fromlist=["_predict"]
    )._predict
    globals()["_collect"] = __import__(
        "giems_lstm.utils", fromlist=["_collect"]
    )._collect


# ==============================================================================
# CLI Commands
# ==============================================================================
def _uniform_entry(debug: bool, parallel: int, seed: int):
    _init_imports()

    _setup_global_logging(debug, "Main")
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
    left: bool = typer.Option(
        False,
        "--left",
        "-l",
        help="If set, this thread will create .pth files first before training starts, then train from ending to beginning. This command will cover --thread-id and --parallel options.",
    ),
):
    """
    Train GIEMS-LSTM models for all locations specified in the config mask.
    """

    _uniform_entry(debug, parallel, seed)
    _train(thread_id, config_path, debug, parallel, left)


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
    eval: bool = typer.Option(
        False, "--eval", "-e", help="Collect evaluation results instead of predictions"
    ),
    parallel: int = typer.Option(
        0, "--parallel", "-p", help="Number of threads to use for collection."
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
    _collect(config_path, eval, parallel)


if __name__ == "__main__":
    app()
