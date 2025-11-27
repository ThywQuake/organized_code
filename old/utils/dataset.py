from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Optional


class WetlandDataset(Dataset):
    def __init__(
        self,
        lat_idx: int,
        lon_idx: int,
        TVARs: dict[str, xr.DataArray],
        CVARs: dict[str, xr.DataArray],
        seq_length: int = 12,
        target_scaler: Optional[MinMaxScaler] = None,
        feature_scalers: Optional[dict[str, MinMaxScaler]] = None,
        window_size: int = 9,
        start_date: str = "1992-01-01",
        end_date: str = "2020-12-01",
        predict: bool = False,
    ):
        """
        Input:
            lat_idx: Latitude index of the location
            lon_idx: Longitude index of the location
            TVARs: Dictionary of xarray DataArrays **with time series** for each variable
            CVARs: Dictionary of xarray DataArrays **without time series** for each variable
            seq_length: Length of the input sequence for each sample
            target_scaler: Optional pre-fitted MinMaxScaler for the target variable
            feature_scalers: Optional pre-fitted MinMaxScaler for the feature variables
            window_size: Size of the spatial window (must be odd)
            start_date: Start date for data selection (YYYY-MM-DD)
            end_date: End date for data selection (YYYY-MM-DD)

        """
        super(WetlandDataset, self).__init__()

        self.lat_idx, self.lon_idx = lat_idx, lon_idx
        self.seq_length, self.window_size = seq_length, window_size
        self.target_scaler = target_scaler
        self.feature_scalers = feature_scalers

        self.giems2 = TVARs.pop("giems2")
        self.TVARs, self.CVARs = TVARs, CVARs
        self.predict = predict

        self.start_date, self.end_date = start_date, end_date
        self.dates = (
            pd.date_range(start=self.start_date, end=self.end_date, freq="MS")
            .to_pydatetime()
            .tolist()
        )

        self.process_data()

    def process_data(self):
        self.load_data()
        self.normalize_data()
        self.windows, self.dates_seq = self.create_windows()

    def load_data(self):
        lat, lon = self.lat_idx, self.lon_idx
        window_radius = self.window_size // 2
        window_lats_raw = np.arange(lat - window_radius, lat + window_radius + 1)
        window_lons_raw = np.arange(lon - window_radius, lon + window_radius + 1)
        # dataset sizes
        lat_size = int(self.giems2.sizes["lat"])  # expected 720
        lon_size = int(self.giems2.sizes["lon"])  # expected 1440
        # apply latitude clamping (no wrap at poles)
        window_lats = np.clip(window_lats_raw, 0, lat_size - 1)
        # apply longitude wrap-around using modulo
        window_lons = np.mod(window_lons_raw, lon_size)
        # mask to null-out values that were originally out-of-range in latitude
        window_mask = np.zeros((self.window_size, self.window_size), dtype=bool)
        window_mask[
            np.ix_(
                (0 <= window_lats_raw) & (window_lats_raw <= lat_size - 1),
                np.ones_like(window_lons_raw, dtype=bool),
            )
        ] = True  # only mask latitudes, longitudes always wrap

        def extract_window(data, T=True):
            raw = data.isel(lat=window_lats, lon=window_lons)
            if T:
                raw = raw.sel(time=slice(self.start_date, self.end_date)).values
            raw = np.where(window_mask, raw, np.nan)
            if not T:
                raw = np.full((len(self.dates),) + raw.shape, raw)
            raw = raw.reshape(len(self.dates), -1)
            return raw

        features = {
            name: extract_window(self.TVARs[name], T=True) for name in self.TVARs
        }
        features.update(
            {name: extract_window(self.CVARs[name], T=False) for name in self.CVARs}
        )

        giems2_center = (
            self.giems2.isel(lat=lat, lon=lon)
            .sel(time=slice(self.start_date, self.end_date))
            .values
        )
        potential = np.full((len(self.dates), 1), np.nanmax(giems2_center))
        features["potential"] = potential
        self.features = {
            name: np.nan_to_num(data, nan=0.0) for name, data in features.items()
        }
        self.target = giems2_center.reshape(-1, 1)

    def normalize_data(self):
        # Feature scalers
        feature_scaled = {}
        if self.feature_scalers:
            for name, data in self.features.items():
                scaler_group = self.feature_scalers[name]
                scaled_data = scaler_group.transform(data)
                feature_scaled[name] = scaled_data
        else:
            feature_scalers = {}
            for name, data in self.features.items():
                scaler_group = MinMaxScaler()
                scaled_data = scaler_group.fit_transform(data)
                feature_scaled[name] = scaled_data
                feature_scalers[name] = scaler_group
            self.feature_scalers = feature_scalers

        # Target scaler (kept in self.scaler for downstream inverse_transform)
        # Handle NaNs in target: fit on non-NaN targets, keep NaNs as NaN after transform
        target_values = self.target.reshape(-1, 1)
        valid_mask = ~np.isnan(target_values.flatten())
        target_scaled = target_values.copy()
        # keep NaNs where original targets were NaN
        if self.target_scaler:
            target_scaled[valid_mask] = self.target_scaler.transform(
                target_values[valid_mask].reshape(-1, 1)
            )
        else:
            target_scaler = MinMaxScaler()
            target_scaler.fit(target_values[valid_mask].reshape(-1, 1))
            target_scaled[valid_mask] = target_scaler.transform(
                target_values[valid_mask].reshape(-1, 1)
            )
            self.target_scaler = target_scaler

        self.features = np.hstack(list(feature_scaled.values()))
        self.target = target_scaled

    def create_windows(self):
        windows = []
        dates_seq = []
        features = self.features
        target = self.target
        date = self.dates

        num_windows = len(target) - self.seq_length + 1
        if self.predict:
            num_windows = len(date) - self.seq_length + 1
        for i in range(num_windows):
            feature_window = features[i : i + self.seq_length]
            window_date = date[i + self.seq_length - 1]
            window = feature_window

            if not self.predict:
                target_index = i + self.seq_length - 1
                if np.isnan(target[target_index]).any():
                    continue
                target_window = target[i + self.seq_length - 1]
                window = (feature_window, target_window)

            windows.append(window)
            dates_seq.append(window_date)

        return windows, dates_seq

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


def wetland_dataloader(
    dataset: WetlandDataset,
    train_years: list[int],
    batch_size: int = 32,
    num_workers: int = 0,
):
    date_years = np.array([date.year for date in dataset.dates_seq])
    selected_indices = np.where(np.isin(date_years, train_years))[0]

    train_subset = Subset(dataset, selected_indices)
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_subset = Subset(dataset, np.where(~np.isin(date_years, train_years))[0])
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader
