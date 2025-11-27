from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple


class WetlandDataset(Dataset):
    """
    A dataset class that creates sequence windows from scaled features and target.
    """

    def __init__(
        self,
        features_scaled: np.ndarray,
        dates: List[pd.Timestamp],
        target_scaled: Optional[np.ndarray] = None,
        seq_length: int = 12,
        predict_mode: bool = False,
    ):
        """
        Input:
            features_scaled: Scaled feature matrix [T, D]
            target_scaled: Scaled target vector [T, 1] (required in training mode)
            dates: Corresponding list of timestamps
            seq_length: Length of input sequences
            predict_mode: Whether in prediction mode
        """
        super().__init__()
        self.features = features_scaled
        self.target = target_scaled
        self.dates = dates
        self.seq_length = seq_length
        self.predict_mode = predict_mode

        self.windows, self.dates_seq = self._create_windows()

    def _create_windows(self) -> Tuple[List, List]:
        """
        Create sequence windows from features and target.
        """
        windows = []
        dates_seq = []

        num_windows = len(self.dates) - self.seq_length + 1
        for i in range(num_windows):
            feature_window = self.features[i : i + self.seq_length]
            window_date = self.dates[i + self.seq_length - 1]

            if not self.predict_mode:
                target_index = i + self.seq_length - 1

                # Skip NaN target values (this logic remains unchanged to ensure training on valid samples only)
                if np.isnan(self.target[target_index]).any():
                    continue

                target_window = self.target[target_index]
                windows.append((feature_window, target_window))
            else:
                windows.append(feature_window)

            dates_seq.append(window_date)

        return windows, dates_seq

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


def wetland_dataloader(
    dataset: WetlandDataset,
    train_years: List[int],
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and testing DataLoaders based on specified training years.
    """

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
