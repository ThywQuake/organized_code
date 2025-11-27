import xarray as xr
import os
import sys
import numpy as np
import torch

from utils.model import LSTMNetKAN
from utils.dataset import WetlandDataset
from utils.pred import Predict

import warnings

warnings.filterwarnings("ignore")

"""============================= Configs ======================================="""

print("Loading data...")
TVARs = {
    "giems2": xr.open_dataset("data/clean/GIEMS-MC_fwet.nc")["fwet"],
    "era5": xr.open_dataset("data/clean/ERA5_tmp.nc")["tmp"],
    "mswep": xr.open_dataset("data/clean/MSWEP_pre.nc")["pre"],
    "gleam": xr.open_dataset("data/clean/GLEAM4a_sm.nc")["sm"],
    "grace": xr.open_dataset("data/clean/GRACE_lwe_thickness.nc")["lwe_thickness"],
}  # TVARs with time series
CVARs = {
    "fcti": xr.open_dataset("data/clean/fcti.nc")["fcti"],
}  # CVARs without time series
mask = xr.open_dataset("data/wetland_mask.nc")["mask"].values

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

total_tasks = int(np.sum(mask))
print(f"Total training locations: {total_tasks}")
tasks_per_thread = 10
# temporary_thread = sys.argv[1]  # e.g., '0'
temporary_thread = '0'

start_task = int(temporary_thread) * tasks_per_thread
end_task = min((int(temporary_thread) + 1) * tasks_per_thread, total_tasks)

hidden_dim = 384
n_layers = 2
n_epochs = 150
verbose_epoch = 100
patience = 50
batch_size = 16
seq_length = 12
window_size = 9
start_date = "1984-01-01"
end_date = "2023-12-01"

model_folder = "output/model/E/"
eval_folder = "output/eval/E/"
pred_folder = "output/pred/E/"

"""============================= Predicting ======================================="""

print("Starting prediction...")
task_counter = 0
for lat_idx in range(mask.shape[0]):
    for lon_idx in range(mask.shape[1]):
        if not mask[lat_idx, lon_idx]:
            continue

        if task_counter < start_task:
            task_counter += 1
            continue
        elif task_counter >= end_task:
            break

        task_counter += 1
        print(f"Processing task {task_counter}/{total_tasks} at ({lat_idx}, {lon_idx})")

        os.makedirs(pred_folder, exist_ok=True)
        save_path = os.path.join(pred_folder, f"{lat_idx}_{lon_idx}.npy")
        if os.path.exists(save_path):
            print(f"Predictions for location ({lat_idx}, {lon_idx}) already exist. Skipping...")
            continue

        dataset = WetlandDataset(
            TVARs=TVARs.copy(),
            CVARs=CVARs,
            lat_idx=lat_idx,
            lon_idx=lon_idx,
            seq_length=seq_length,
            window_size=window_size,
            start_date=start_date,
            end_date=end_date,
        )
        input_dim = (len(TVARs) + len(CVARs) - 1) * (window_size**2) + 1

        model = LSTMNetKAN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            n_layers=n_layers,
            device=device
        )

        model_path = os.path.join(model_folder, f"{lat_idx}_{lon_idx}.pth")
        if not os.path.exists(model_path):
            print(f"Model file {model_path} does not exist. Skipping...")
            continue

        model.load_state_dict(torch.load(model_path, map_location=device))

        pred = Predict(
            dataset=dataset,
            lat_idx=lat_idx,
            lon_idx=lon_idx,
            start_date=start_date,
            end_date=end_date,
            model=model,
            device=device,
            batch_size=batch_size,
            save_path=save_path,
        )

        pred.run()
