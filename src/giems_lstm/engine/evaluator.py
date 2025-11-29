import numpy as np
import torch
import os
import logging
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Union

from ..model import LSTMNetKAN, LSTMNet, GRUNetKAN, GRUNet

# Configure logging
logger = logging.getLogger()
if not logger.handlers:
    # Add StreamHandler if no configuration is found
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)


# ==============================================================================
# METRICS UTILITIES (Separated from the main class for better structure)
# ==============================================================================


def sMAPE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Symmetric Mean Absolute Percentage Error (sMAPE)."""
    # sMAPE formula: 100 * mean(|y_pred - y_true| / (|y_true| + |y_pred|)/2)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero by setting 0/0 to 0 (where both true and pred are 0)
    ratio = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )

    # Keeping the original calculation format for fidelity, but note the division by 2 is unusual.
    return (100.0 / len(y_true)) * np.sum(ratio)


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Nash-Sutcliffe Efficiency (NSE)."""
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    # Handle division by zero if all y_true values are the same (denominator=0)
    if denominator == 0:
        return 1.0 if numerator == 0 else -float("inf")
    return 1 - (numerator / denominator)


# ==============================================================================
# EVALUATOR CLASS
# ==============================================================================


class Evaluator:
    """
    Evaluates a trained sequence model (LSTM/GRU/KAN) by running inference,
    inverse scaling predictions, and calculating standard hydrological and ML metrics.
    """

    def __init__(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        lat_idx: int,
        lon_idx: int,
        model: Union[LSTMNetKAN, LSTMNet, GRUNetKAN, GRUNet],
        eval_folder: str,
        target_scaler: MinMaxScaler,
        device: torch.device,
        debug: bool = False,
    ):
        """
        Initializes the evaluation process.

        Args:
            train_dataloader (DataLoader): DataLoader for training data.
            test_dataloader (DataLoader): DataLoader for testing data.
            lat_idx (int): Latitude index of the location being evaluated.
            lon_idx (int): Longitude index of the location being evaluated.
            model (Union[LSTMNetKAN, LSTMNet, GRUNetKAN, GRUNet]): The trained model to evaluate.
            eval_folder (str): Directory to save evaluation results.
            target_scaler (MinMaxScaler): Scaler used for inverse transforming target values.
            device (torch.device): Device to run the model on (CPU/GPU).
            debug (bool): If True, forces re-evaluation even if results exist.
        """
        self.model = model
        self.lat_idx = lat_idx
        self.lon_idx = lon_idx
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader
        self.eval_folder = eval_folder
        self.scaler = target_scaler
        self.device = device

        # Internal storage for results
        self.Y_scaled: Dict[str, List[np.ndarray]] = {}
        self.Y_inv: Dict[str, np.ndarray] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.debug = debug

    def run(self):
        """
        Main evaluation workflow: loads model, runs inference, calculates metrics, and saves results.
        """
        os.makedirs(self.eval_folder, exist_ok=True)
        save_path = os.path.join(self.eval_folder, f"{self.lat_idx}_{self.lon_idx}.npy")
        if not self.debug and os.path.exists(save_path):
            logger.info(
                f"Evaluation for location ({self.lat_idx}, {self.lon_idx}) already exists at {save_path}. Skipping."
            )
            return

        logger.info(f"Processing location ({self.lat_idx}, {self.lon_idx})...")
        self.model.eval()

        # Run the core evaluation steps
        self._run_inference()
        self._inverse_transform_results()
        self._calculate_metrics()

        # Save results to disk
        self._save_results()

    def _run_inference(self):
        """
        Runs inference over both train and test data loaders and collects scaled predictions
        and true values into self.Y_scaled.
        """
        logger.debug("  Running inference...")

        data_loaders = {
            "train": self.train_loader,
            "test": self.test_loader,
        }

        self.Y_scaled = {
            "y_pred_train": [],
            "y_true_train": [],
            "y_pred_test": [],
            "y_true_test": [],
        }

        with torch.no_grad():
            for split_name, loader in data_loaders.items():
                # Check if loader is empty before proceeding
                if len(loader) == 0:
                    logger.warning(
                        f"  {split_name} dataloader is empty. Skipping inference for this split."
                    )
                    continue

                for inputs, targets in loader:
                    inputs = inputs.to(self.device).float()
                    # Targets do not need to be moved to GPU for loss, but must be available for prediction/true value collection

                    # Initialize hidden state
                    # NOTE: This part needs the model's init_hidden logic, which is usually separate
                    # For simplicity, we assume the model has a working init_hidden method:
                    h = self.model.init_hidden(inputs.size(0))

                    outputs, h_out = self.model(inputs, h)

                    # Store results on CPU
                    self.Y_scaled[f"y_pred_{split_name}"].append(
                        outputs.cpu().numpy().flatten()
                    )
                    self.Y_scaled[f"y_true_{split_name}"].append(
                        targets.cpu().numpy().flatten()
                    )

    def _inverse_transform_results(self):
        """
        Concatenates scaled results and inverse transforms them to the original scale.
        """
        logger.info("  Inverse transforming results...")

        def inverse_transform_single(scaled_array_list: List[np.ndarray]) -> np.ndarray:
            """Handles concatenation and inverse transformation for one array set."""
            if not scaled_array_list:
                return np.array([])

            y = np.concatenate(scaled_array_list).reshape(-1, 1)
            y_inv = self.scaler.inverse_transform(y)

            # Original logic: set small negative values to zero
            y_inv[y_inv < 0.0003] = 0.0
            return y_inv

        self.Y_inv = {
            name: inverse_transform_single(self.Y_scaled[name])
            for name in self.Y_scaled
            if name.startswith("y_")
        }

    def _calculate_metrics(self):
        """
        Calculates R2, RMSE, sMAPE, and NSE for both train and test splits.
        """
        logger.info("  Calculating metrics...")

        self.metrics = {}

        for split in ["train", "test"]:
            true_key = f"y_true_{split}"
            pred_key = f"y_pred_{split}"

            y_true_inv = self.Y_inv.get(true_key)
            y_pred_inv = self.Y_inv.get(pred_key)

            if y_true_inv is None or y_true_inv.size == 0:
                self.metrics[split] = {"status": "Skipped (No Data)"}
                continue

            self.metrics[split] = {
                "sMAPE": sMAPE(y_true_inv, y_pred_inv),
                "NSE": nse(y_true_inv, y_pred_inv),
                "R2": r2_score(y_true_inv, y_pred_inv),
                "RMSE": np.sqrt(mean_squared_error(y_true_inv, y_pred_inv)),
            }

            logger.debug(
                f"    {split.upper()} Metrics: R2={self.metrics[split]['R2']:.4f}, RMSE={self.metrics[split]['RMSE']:.4f}"
            )

    def _save_results(self):
        """
        Saves the inverse transformed results (predictions/true values) and calculated metrics.
        """
        eval_path = os.path.join(self.eval_folder, f"{self.lat_idx}_{self.lon_idx}.npy")

        # Convert NumPy arrays in Y_inv to simple dict for saving
        # Metrics dict is already suitable

        np.save(
            eval_path,
            {
                "Y_inv": self.Y_inv,
                "metrics": self.metrics,
            },
            # Allow saving of dicts containing objects (like dicts/lists)
            allow_pickle=True,
        )
        logger.info(f"Evaluation results saved to {eval_path}")
