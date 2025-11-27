import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from utils.model import LSTMNetKAN
from utils.dataset import WetlandDataset, wetland_dataloader


class Eval:
    def __init__(
        self,
        model: LSTMNetKAN,
        lat_idx: int,
        lon_idx: int,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        eval_folder: str,
        model_folder: str,
        target_scaler: MinMaxScaler,
        device: torch.device,
    ):
        """
        Initialize the evaluation process with model, data loaders, scaler, and dataset.

        Args:
            model (LSTMNetKAN): The trained model to evaluate.
            lat_idx (int): Latitude index of the location.
            lon_idx (int): Longitude index of the location.
            train_dataloader (DataLoader): DataLoader for training data.
            test_dataloader (DataLoader): DataLoader for testing data.
            eval_folder (str): Folder path to save evaluation results.
            model_folder (str): Folder path where the trained model is saved.
            target_scaler (MinMaxScaler): Scaler used for the target variable.
            device (torch.device): Device to run the evaluation on.
        """
        self.model = model
        self.lat_idx = lat_idx
        self.lon_idx = lon_idx
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader
        self.eval_folder = eval_folder
        self.model_folder = model_folder
        self.scaler = target_scaler
        self.device = device

    def run(self):
        if not os.path.exists(self.eval_folder):
            os.makedirs(self.eval_folder)

        model_path = os.path.join(
            self.model_folder, f"{self.lat_idx}_{self.lon_idx}.pth"
        )
        if not os.path.exists(model_path):
            print(
                f"Model for location ({self.lat_idx}, {self.lon_idx}) not found. Skipping."
            )
            return
        print(f"Processing location ({self.lat_idx}, {self.lon_idx})...")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.eval()

        eval_path = os.path.join(self.eval_folder, f"{self.lat_idx}_{self.lon_idx}.npy")
        np.save(
            eval_path,
            {
                "Y_inv": self.Y_inv,
                "metrics": self.metrics,
            },
        )

    @staticmethod
    def sMAPE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Symmetric Mean Absolute Percentage Error (sMAPE).

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The sMAPE value.
        """
        return (
            (100.0 / len(y_true))
            * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
            / 2
        )

    @staticmethod
    def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Nash-Sutcliffe Efficiency (NSE).

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The NSE value.
        """
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (numerator / denominator)

    def inverse(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform the scaled predictions to original scale.

        Args:
            y_scaled (np.ndarray): Scaled predictions.

        Returns:
            np.ndarray: Predictions in original scale.
        """
        y = np.concatenate(y_scaled).reshape(-1, 1)
        y_inv = self.scaler.inverse_transform(y)
        y_inv[y_inv < 0.0003] = 0.0  # Set small negative values to zero
        return y_inv

    def eval(self):
        self.model.eval()
        Y = {
            "y_pred_train": [],
            "y_true_train": [],
            "y_pred_test": [],
            "y_true_test": [],
        }

        train_y_trues = []
        for inputs, targets in self.train_loader:
            train_y_trues.append(targets.cpu().numpy().reshape(-1))
        train_y_trues = np.concatenate(train_y_trues)

        with torch.no_grad():
            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device).float()
                targets = targets.to(self.device).float()

                h = self.model.init_hidden(inputs.size(0))
                outputs, h = self.model(inputs, h)
                Y["y_pred_train"].append(outputs.cpu().numpy().flatten())
                Y["y_true_train"].append(targets.cpu().numpy().flatten())

            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device).float()
                targets = targets.to(self.device).float()

                h = self.model.init_hidden(inputs.size(0))
                outputs, h = self.model(inputs, h)
                Y["y_pred_test"].append(outputs.cpu().numpy().flatten())
                Y["y_true_test"].append(targets.cpu().numpy().flatten())

        Y_inv = {name: self.inverse(Y[name]) for name in Y}

        self.Y_inv = Y_inv
        self.metrics = {
            "train": {
                "sMAPE": self.sMAPE(Y_inv["y_true_train"], Y_inv["y_pred_train"]),
                "NSE": self.nse(Y_inv["y_true_train"], Y_inv["y_pred_train"]),
                "R2": r2_score(Y_inv["y_true_train"], Y_inv["y_pred_train"]),
                "RMSE": np.sqrt(
                    mean_squared_error(Y_inv["y_true_train"], Y_inv["y_pred_train"])
                ),
            },
            "test": {
                "sMAPE": self.sMAPE(Y_inv["y_true_test"], Y_inv["y_pred_test"]),
                "NSE": self.nse(Y_inv["y_true_test"], Y_inv["y_pred_test"]),
                "R2": r2_score(Y_inv["y_true_test"], Y_inv["y_pred_test"]),
                "RMSE": np.sqrt(
                    mean_squared_error(Y_inv["y_true_test"], Y_inv["y_pred_test"])
                ),
            },
        }
