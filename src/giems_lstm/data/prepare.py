import numpy as np
import xarray as xr
import pandas as pd
from typing import Dict, Tuple, List


def extract_window_data(
    lat_idx: int,
    lon_idx: int,
    TVARs: Dict[str, xr.DataArray],
    CVARs: Dict[str, xr.DataArray],
    window_size: int,
    start_date: str,
    end_date: str,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[pd.Timestamp]]:
    """
    Load raw data for a specific location and extract a spatial window

    Args:
        lat_idx, lon_idx: Latitude and Longitude indices of the target location.
        TVARs, CVARs: Dictionaries of time-varying and constant xarray variables.
        window_size: Spatial window size.
        start_date, end_date: Time range.

    Returns:
        (features: Dict[str, np.ndarray], target: np.ndarray, dates: List[pd.Timestamp]):
        Raw (unscaled) feature matrix, target vector, and list of timestamps.
    """

    window_radius = window_size // 2
    window_lats_raw = np.arange(lat_idx - window_radius, lat_idx + window_radius + 1)
    window_lons_raw = np.arange(lon_idx - window_radius, lon_idx + window_radius + 1)
    # dataset sizes
    lat_size = int(TVARs["giems2"].sizes["lat"])  # expected 720
    lon_size = int(TVARs["giems2"].sizes["lon"])  # expected 1440
    # apply latitude clamping (no wrap at poles)
    window_lats = np.clip(window_lats_raw, 0, lat_size - 1)
    # apply longitude wrap-around using modulo
    window_lons = np.mod(window_lons_raw, lon_size)
    # mask to null-out values that were originally out-of-range in latitude
    window_mask = np.zeros((window_size, window_size), dtype=bool)
    window_mask[
        np.ix_(
            (0 <= window_lats_raw) & (window_lats_raw <= lat_size - 1),
            np.ones_like(window_lons_raw, dtype=bool),
        )
    ] = True  # only mask latitudes, longitudes always wrap

    def extract_window(data, T=True):
        raw = data.isel(lat=window_lats, lon=window_lons)
        if T:
            raw = raw.sel(time=slice(start_date, end_date)).values
        raw = np.where(window_mask, raw, np.nan)
        if not T:
            raw = np.full((len(dates),) + raw.shape, raw)
        raw = raw.reshape(len(dates), -1)
        return raw

    dates = (
        pd.date_range(start=start_date, end=end_date, freq="MS")
        .to_pydatetime()
        .tolist()
    )

    features = {}
    for name, data_array in TVARs.items():
        if name == "giems2":
            continue  # skip target variable here
        features[name] = extract_window(data_array, T=True)

    for name, data_array in CVARs.items():
        features[name] = extract_window(data_array, T=False)

    giems2_center = (
        TVARs["giems2"]
        .isel(lat=lat_idx, lon=lon_idx)
        .sel(time=slice(start_date, end_date))
        .values
    )
    potential = np.full((len(dates), 1), np.nanmax(giems2_center))
    features["potential"] = potential

    features = {name: np.nan_to_num(data, nan=0.0) for name, data in features.items()}
    target = giems2_center.reshape(-1, 1)

    return features, target, dates
