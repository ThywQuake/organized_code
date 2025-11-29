import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import os
import logging
from typing import Union
from statsmodels.tsa.ar_model import AutoReg

from ..data import WetlandDataset
from ..model import LSTMNetKAN, LSTMNet, GRUNetKAN, GRUNet


# Configure logging
logger = logging.getLogger()
if not logger.handlers:
    # Add StreamHandler if no configuration is found
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)


class Predictor:
    """
    Handles the entire prediction workflow: inference on the dataset, inverse scaling
    using the TRAIN SCALER, optional backcasting, and saving the results.
    """

    def __init__(
        self,
        lat_idx: int,
        lon_idx: int,
        dataset: WetlandDataset,
        target_scaler: MinMaxScaler,
        model: Union[LSTMNetKAN, LSTMNet, GRUNetKAN, GRUNet],
        save_path: str,
        device: torch.device,
        batch_size: int = 64,
        debug: bool = False,
    ):
        """
        Initializes the predictor with location indices, dataset, model, and settings.

        Args:
            lat_idx (int): Latitude index of the location to predict.
            lon_idx (int): Longitude index of the location to predict.
            dataset (WetlandDataset): Dataset containing windows for prediction.
            target_scaler (MinMaxScaler): Scaler used for inverse transforming target values.
            model (Union[LSTMNetKAN, LSTMNet, GRUNetKAN, GRUNet]): The trained model for prediction.
            save_path (str): File path to save the prediction results.
            device (torch.device): Device to run the model on (CPU/GPU).
            batch_size (int): Batch size for model inference.
            debug (bool): If True, forces re-prediction even if results exist.
        """
        self.lat_idx = lat_idx
        self.lon_idx = lon_idx
        self.save_path = save_path
        self.device = device
        self.dataset = dataset
        self.target_scaler = target_scaler
        self.model = model
        self.batch_size = batch_size
        self.debug = debug
        self.model.to(self.device)
        self.pred_scaled: np.ndarray = np.array([])
        self.pred_final: np.ndarray = np.array([])

    def run(self):
        """
        Main entry point for the prediction process.
        """
        logger.info(
            f"Starting prediction for location ({self.lat_idx}, {self.lon_idx})..."
        )

        if not self.debug and os.path.exists(self.save_path):
            logger.info(
                f"Prediction file already exists at {self.save_path}. Skipping."
            )
            return

        # Core workflow steps
        self._run_inference()
        self._post_process_predictions()
        self._backfill()
        self._save_results()

        logger.debug(
            f"Prediction completed for location ({self.lat_idx}, {self.lon_idx})."
        )

    def _run_inference(self):
        """
        Runs the trained model on all prediction windows and collects scaled outputs.
        """
        logger.debug("  Running model inference...")

        # NOTE: The dataset must expose windows (np.ndarray of features)
        windows = self.dataset.windows

        if not windows:
            logger.warning("  Dataset windows are empty. Cannot run inference.")
            return

        self.model.eval()
        predictions_list = []

        with torch.no_grad():
            for i in range(0, len(windows), self.batch_size):
                # Convert list of NumPy arrays to a torch Tensor batch
                batch = torch.tensor(
                    np.array(windows[i : i + self.batch_size]), dtype=torch.float32
                ).to(self.device)

                # Initialize hidden state for the batch
                h = self.model.init_hidden(batch.size(0))

                output, _ = self.model(batch, h)

                # Detach, move to CPU, convert to numpy, and flatten
                pred = output.detach().cpu().numpy().reshape(-1)
                predictions_list.append(pred)

        self.pred_scaled = np.concatenate(predictions_list)
        logger.debug(
            f"  Inference finished. Total predictions: {len(self.pred_scaled)}"
        )

    def _post_process_predictions(self):
        """
        Inverse transforms the scaled predictions using the TRAIN SCALER
        and applies thresholding.
        """
        if self.pred_scaled.size == 0:
            return

        # CRITICAL: Use target_scaler from the dataset, which was fitted on the training set
        target_scaler: MinMaxScaler = self.target_scaler

        if target_scaler is None:
            logger.error(
                "Target scaler not available in dataset. Cannot inverse transform."
            )
            return

        predictions_rescaled = target_scaler.inverse_transform(
            self.pred_scaled.reshape(-1, 1)
        ).reshape(-1)

        # Thresholding: Set small negative values to zero
        predictions_rescaled[predictions_rescaled < 0.0003] = 0.0

        self.pred_final = predictions_rescaled
        logger.debug(
            "  Predictions successfully rescaled and thresholded using TRAIN SCALER."
        )

    def _backfill(self):
        """
        Applies AutoRegressive (AR) modeling to estimate missing values
        in the prediction lead-in period (backcasting).
        """
        if self.pred_final.size == 0:
            logger.warning("  No final predictions to backfill.")
            return

        logger.debug("  Starting AutoRegressive backfilling...")
        try:
            # 1. Prepare data: Replace NaNs with the mean for robust AR fitting
            series = np.nan_to_num(
                self.pred_final, nan=float(np.nanmean(self.pred_final))
            )

            # 2. Fit AR model: using 12 lags
            ar_model = AutoReg(series, lags=12)
            ar_fit = ar_model.fit()

            # 3. Determine backcast length (seq_length - 1)
            backcast_len = self.dataset.seq_length - 1

            if backcast_len <= 0:
                logger.warning("  Sequence length is 1, skipping backfill.")
                return

            # 4. Backcast: Predict the first 'backcast_len' values of the full period
            # NOTE: The AutoReg index handling is complex, this logic aims to prepend the necessary sequence.
            # Start prediction index is 1 - seq_length, End is -1 (predicts up to the first actual prediction)
            backcast = ar_fit.predict(
                start=len(series) - backcast_len, end=len(series) - 1
            )

            # Thresholding for backcast
            backcast[backcast < 0.0003] = 0.0

            # 5. Concatenate backcast with main predictions
            self.pred_final = np.concatenate([backcast, self.pred_final])
            logger.debug(f"  Backfill successful. Added {len(backcast)} steps.")

        except Exception as e:
            logger.error(
                f"  Backfilling failed for ({self.lat_idx}, {self.lon_idx}): {e}"
            )

    def _save_results(self):
        """
        Saves the final predictions and corresponding dates to a NumPy file.
        The final prediction array (self.pred_final) is expected to match the length
        of the full date range (self.dataset.dates) after backfilling.
        """
        if self.pred_final.size == 0:
            logger.warning("  No final results to save.")
            return

        eval_path = self.save_path

        # Verification of Date Alignment (Final Check)
        if len(self.dataset.dates) != len(self.pred_final):
            logger.error(
                f"Date length ({len(self.dataset.dates)}) does not match final prediction length ({len(self.pred_final)}) after backfill. Saving predictions without dates to avoid misalignment."
            )
            # This is an error state, save only predictions
            preds_to_save = {"prediction": self.pred_final}
        else:
            # This is the desired state: full dates + full predictions
            preds_to_save = {"date": self.dataset.dates, "prediction": self.pred_final}

        np.save(eval_path, preds_to_save, allow_pickle=True)
        logger.info(f"Predictions successfully saved to {eval_path}")
