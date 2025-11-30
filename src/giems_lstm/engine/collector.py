import logging
import numpy as np
import pandas as pd
from pathlib import Path
import xarray as xr

from giems_lstm.config import Config
from giems_lstm.utils.allocate_coords import _allocate_coords


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


class Collector:
    def __init__(self, config: Config, eval: bool):
        """
        Initialize the Collector with configuration and mode.
        """

        self.config = config
        self.eval = eval
        self.logger = logging.getLogger()
        self._load_data()
        self._initialize_array()

    def run(self, file_batch: list[Path]):
        self._collect_batch(file_batch)

    def save(self):
        self._save_output()

    def _load_data(self):
        if self.eval:
            self.source_folder = self.config.eval_folder
            self.output_file = self.config.predict.output_file.replace(
                ".nc", "_eval.nc"
            )
        else:
            self.source_folder = self.config.pred_folder
            self.output_file = self.config.predict.output_file

        if not self.source_folder.exists():
            self.logger.error(f"Source folder {self.source_folder} does not exist.")
            raise FileNotFoundError(
                f"Source folder {self.source_folder} does not exist."
            )

        self.logger.debug(f"Scanning files in {self.source_folder}...")
        self.files = list(self.source_folder.glob("*.npy"))
        if not self.files:
            self.logger.error("No .npy files found.")
            raise FileNotFoundError("No .npy files found.")

        GIEMS = self.config.TVARs["giems2"]
        self.lats = GIEMS.coords["lat"].values
        self.lons = GIEMS.coords["lon"].values
        self.dates = pd.date_range(
            start=self.config.predict.start_date,
            end=self.config.predict.end_date,
            freq="MS",
        )
        self.coords = _allocate_coords(self.config.mask, 0, self.config.total_tasks)

    def _initialize_array(self):
        template_zero_array = np.full(
            (len(self.dates), len(self.lats), len(self.lons)), np.nan, dtype=np.float32
        )
        if self.eval:
            self.sMAPE_train = template_zero_array.copy()
            self.sMAPE_test = template_zero_array.copy()
            self.NSE_train = template_zero_array.copy()
            self.NSE_test = template_zero_array.copy()
            self.R2_train = template_zero_array.copy()
            self.R2_test = template_zero_array.copy()
            self.RMSE_train = template_zero_array.copy()
            self.RMSE_test = template_zero_array.copy()

        else:
            self.predictions_array = template_zero_array.copy()

    def _collect_eval_single(self, file_path: Path):
        lat_idx, lon_idx = _parse_filename(file_path)
        data = np.load(file_path, allow_pickle=True).item()["metrics"]
        self.sMAPE_train[:, lat_idx, lon_idx] = data.get("train", {}).get(
            "sMAPE", np.nan
        )
        self.sMAPE_test[:, lat_idx, lon_idx] = data.get("test", {}).get("sMAPE", np.nan)
        self.NSE_train[:, lat_idx, lon_idx] = data.get("train", {}).get("NSE", np.nan)
        self.NSE_test[:, lat_idx, lon_idx] = data.get("test", {}).get("NSE", np.nan)
        self.R2_train[:, lat_idx, lon_idx] = data.get("train", {}).get("R2", np.nan)
        self.R2_test[:, lat_idx, lon_idx] = data.get("test", {}).get("R2", np.nan)
        self.RMSE_train[:, lat_idx, lon_idx] = data.get("train", {}).get("RMSE", np.nan)
        self.RMSE_test[:, lat_idx, lon_idx] = data.get("test", {}).get("RMSE", np.nan)

    def _collect_pred_single(self, file_path: Path):
        lat_idx, lon_idx = _parse_filename(file_path)
        data = np.load(file_path, allow_pickle=True).item()["prediction"]
        self.predictions_array[:, lat_idx, lon_idx] = data

    def _collect_batch(self, file_batch: list[Path]):
        for file_path in file_batch:
            try:
                if self.eval:
                    self._collect_eval_single(file_path)
                else:
                    self._collect_pred_single(file_path)
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")

    def _save_output(self):
        if self.eval:
            ds = xr.Dataset(
                {
                    "sMAPE_train": (("time", "lat", "lon"), self.sMAPE_train),
                    "sMAPE_test": (("time", "lat", "lon"), self.sMAPE_test),
                    "NSE_train": (("time", "lat", "lon"), self.NSE_train),
                    "NSE_test": (("time", "lat", "lon"), self.NSE_test),
                    "R2_train": (("time", "lat", "lon"), self.R2_train),
                    "R2_test": (("time", "lat", "lon"), self.R2_test),
                    "RMSE_train": (("time", "lat", "lon"), self.RMSE_train),
                    "RMSE_test": (("time", "lat", "lon"), self.RMSE_test),
                },
                coords={
                    "time": self.dates,
                    "lat": self.lats,
                    "lon": self.lons,
                },
            )
        else:
            ds = xr.Dataset(
                {
                    "fwet": (("time", "lat", "lon"), self.predictions_array),
                },
                coords={
                    "time": self.dates,
                    "lat": self.lats,
                    "lon": self.lons,
                },
            )
        encoding = {
            var: {"zlib": True, "complevel": 5, "_FillValue": np.nan}
            for var in ds.data_vars
        }
        ds.to_netcdf(self.output_file, encoding=encoding)
        self.logger.info(f"Saved collected data to {self.output_file}")
