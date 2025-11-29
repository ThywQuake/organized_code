import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple, Optional

ScalerSet = Tuple[Optional[MinMaxScaler], Optional[Dict[str, MinMaxScaler]]]


def fit_scalers(
    raw_target: np.ndarray,
    raw_features: Dict[str, np.ndarray],
) -> ScalerSet:
    """
    Fit MinMaxScalers for features and target variable.

    Args:
        raw_features: Original feature matrix [T, D]
        raw_target: Original target vector [T, 1]
        feature_names: List of feature names to distinguish feature columns (optional but recommended)

    Returns:
        (target_scaler, feature_scalers): Tuple of fitted scalers
    """

    target_scaler = MinMaxScaler()
    valid_mask = ~np.isnan(raw_target.flatten())
    target_scaler.fit(raw_target[valid_mask].reshape(-1, 1))

    feature_scalers = {}
    for name, data in raw_features.items():
        scaler = MinMaxScaler()
        scaler.fit(data)
        feature_scalers[name] = scaler

    return target_scaler, feature_scalers


def transform_target(
    target: np.ndarray,
    target_scaler: MinMaxScaler,
) -> np.ndarray:
    """
    Transform target variable using the fitted scaler.

    Args:
        target: Original target vector [T, 1]
        target_scaler: Fitted MinMaxScaler for target

    Returns:
        target_scaled: Scaled target vector [T, 1]
    """
    target_values = target.reshape(-1, 1)
    valid_mask = ~np.isnan(target_values.flatten())
    target_scaled = target_values.copy()
    target_scaled[valid_mask] = target_scaler.transform(
        target_values[valid_mask].reshape(-1, 1)
    )
    return target_scaled


def transform_features(
    features: Dict[str, np.ndarray],
    feature_scalers: Dict[str, MinMaxScaler],
) -> np.ndarray:
    """
    Transform feature matrix using the fitted scalers.

    Args:
        features: Original feature matrix as a dict of arrays [T, D_i]
        feature_scalers: Dict of fitted MinMaxScalers for each feature

    Returns:
        features_scaled: Scaled feature matrix [T, D]
    """
    feature_scaled = {}
    for name, data in features.items():
        scaler = feature_scalers[name]
        scaled_data = scaler.transform(data)
        feature_scaled[name] = scaled_data

    features_scaled = np.hstack(list(feature_scaled.values()))
    return features_scaled
