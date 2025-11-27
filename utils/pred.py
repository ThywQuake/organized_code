import numpy as np
import pandas as pd
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
        start_date: str,
        end_date: str,
        model: LSTMNetKAN,
        save_path: str,
        device: torch.device,
        batch_size: int = 64,
    ):
        """
        Initialize the prediction process with the dataset.

        Args:
            dataset (WetlandDataset): The dataset for making predictions.
            start_date (str): The start date for the prediction period.
            end_date (str): The end date for the prediction period. 
            model (LSTMNetKAN): The model to be used for predictions.
            device (torch.device): Device to run the prediction on. 
            save_path (str): The path where predictions will be saved.
        """
        self.lat_idx, self.lon_idx = lat_idx, lon_idx
        self.save_path = save_path
        self.device = device
        self.dataset = dataset
        self.start_date = start_date
        self.end_date = end_date
        self.model = model
        self.batch_size = batch_size
        
    def run(self):
        features_scaler = self.dataset.feature_scalers
        target_scaler = self.dataset.target_scaler
        windows, dates_seq = self.build_windows()
        
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, len(windows), self.batch_size):
                batch = torch.tensor(windows[i:i+self.batch_size], dtype=torch.float32).to(self.device)
                h = self.model.init_hidden(batch.size(0))
                output, _ = self.model(batch, h)
                pred = output.detach().cpu().numpy().reshape(-1)
                predictions.append(pred)
        predictions = np.concatenate(predictions)
        
        predictions_rescaled = target_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1)
        predictions_rescaled[predictions_rescaled < 0.0003] = 0.0  # Set small negative values to zero
        
        self.pred = predictions_rescaled
        self.backfill()
        
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='MS').to_pydatetime().tolist()
        preds = {
            "date": dates,
            "prediction": self.pred
        }
        np.save(self.save_path, preds)
        print(f"Predictions saved to {self.save_path}")
        
    def build_windows(self):
        self.dataset.load_data(start_date=self.start_date, end_date=self.end_date)
        self.dataset.normalize_data()
        return self.dataset.create_windows(predict=True)
    
    def backfill(self):
        try:
            series = np.nan_to_num(self.pred, nan=float(np.nanmean(self.pred)))
            ar_model = AutoReg(series, lags=12)
            ar_fit = ar_model.fit()
            backcast = ar_fit.predict(start=1-self.dataset.seq_length, end=-1)
            backcast[backcast < 0.0003] = 0.0  # Set small negative values to zero
        
        except Exception as e:
            print(f"Backfilling failed: {e}")
            
        self.pred = np.concatenate([backcast, self.pred])