import numpy as np
import torch
from statsmodels.tsa.ar_model import AutoReg

from utils.dataset import WetlandDataset
from utils.model import LSTMNetKAN


class Predict:
    def __init__(
        self,
        lat_idx: int,
        lon_idx: int,
        dataset: WetlandDataset,
        model: LSTMNetKAN,
        save_path: str,
        device: torch.device,
        batch_size: int = 64,
    ):
        """
        Initialize the prediction process with the dataset.

        Args:
            dataset (WetlandDataset): The dataset for making predictions.
            model (LSTMNetKAN): The model to be used for predictions.
            device (torch.device): Device to run the prediction on.
            save_path (str): The path where predictions will be saved.
        """
        self.lat_idx, self.lon_idx = lat_idx, lon_idx
        self.save_path = save_path
        self.device = device
        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size

    def run(self):
        target_scaler = self.dataset.target_scaler
        windows = self.dataset.windows

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, len(windows), self.batch_size):
                batch = torch.tensor(
                    windows[i : i + self.batch_size], dtype=torch.float32
                ).to(self.device)
                h = self.model.init_hidden(batch.size(0))
                output, _ = self.model(batch, h)
                pred = output.detach().cpu().numpy().reshape(-1)
                predictions.append(pred)
        predictions = np.concatenate(predictions)

        predictions_rescaled = target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).reshape(-1)
        predictions_rescaled[predictions_rescaled < 0.0003] = (
            0.0  # Set small negative values to zero
        )

        self.pred = predictions_rescaled
        self.backfill()

        dates = self.dataset.dates
        preds = {"date": dates, "prediction": self.pred}
        np.save(self.save_path, preds)
        print(f"Predictions saved to {self.save_path}")

    def backfill(self):
        try:
            series = np.nan_to_num(self.pred, nan=float(np.nanmean(self.pred)))
            ar_model = AutoReg(series, lags=12)
            ar_fit = ar_model.fit()
            backcast = ar_fit.predict(start=1 - self.dataset.seq_length, end=-1)
            backcast[backcast < 0.0003] = 0.0  # Set small negative values to zero

        except Exception as e:
            print(f"Backfilling failed: {e}")

        self.pred = np.concatenate([backcast, self.pred])
