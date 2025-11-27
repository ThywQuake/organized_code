import xarray as xr
import os
import sys
import numpy as np
import torch

from utils.model import LSTMNetKAN
from utils.dataset import WetlandDataset, wetland_dataloader
from utils.train import Train
from utils.eval import Eval

import warnings
warnings.filterwarnings("ignore")

'''============================= Configs ======================================='''

print("Loading data...")
TVARs = {
    "giems2": xr.open_dataset("data/clean/GIEMS-MC_fwet.nc")["fwet"],
    "era5": xr.open_dataset("data/clean/ERA5_tmp.nc")["tmp"],
    "mswep": xr.open_dataset("data/clean/MSWEP_pre.nc")["pre"],
    "gleam": xr.open_dataset("data/clean/GLEAM4a_sm.nc")["sm"],
    "grace": xr.open_dataset("data/clean/GRACE_lwe_thickness.nc")["lwe_thickness"]
} # TVARs with time series
CVARs = {
    "fcti": xr.open_dataset("data/clean/fcti.nc")["fcti"],
} # CVARs without time series
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


train_years = [1992,1993,1994,2003,2004,2005,2007,2008,2009,2010,2011,2012,2013,2014,2018,2019]
lr = 0.0005
hidden_dim = 384
n_layers = 2
n_epochs = 150
verbose_epoch = 100
patience = 50
batch_size = 16
seq_length = 12
window_size = 9
start_date = "1992-01-01"
end_date = "2020-12-01"

model_folder = "output/model/E/"
eval_folder = "output/eval/E/"

'''============================= Training & Evaluation ======================================='''

lat_idxs, lon_idxs = [], []  # To store lat/lon indices for evaluation later
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
        lat_idxs.append(lat_idx)
        lon_idxs.append(lon_idx)
        print(f"Processing location {task_counter}/{total_tasks} at lat_idx: {lat_idx}, lon_idx: {lon_idx}")
        
        os.makedirs(model_folder, exist_ok=True)
        model_path = os.path.join(model_folder, f"{lat_idx}_{lon_idx}.pth")
        if os.path.exists(model_path):
            print(f"Model for location at lat_idx: {lat_idx}, lon_idx: {lon_idx} already exists. Skipping training.")
            continue
        
        dataset = WetlandDataset(
            lat_idx=lat_idx,
            lon_idx=lon_idx,
            TVARs=TVARs.copy(),
            CVARs=CVARs,
            seq_length=seq_length,
            window_size=window_size,
            start_date=start_date,
            end_date=end_date,
        )
        train_loader, test_loader = wetland_dataloader(
            dataset=dataset,
            train_years=train_years,
        )
        trainer = Train(
            train_loader=train_loader,
            test_loader=test_loader,
            learn_rate=lr,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_epochs=n_epochs,
            model_type='LSTM_KAN',
            verbose_epoch=verbose_epoch,
            device=device,
            patience=patience,
        )
        model = trainer.run()
        if model is None:
            print(f"Training failed for location at lat_idx: {lat_idx}, lon_idx: {lon_idx}. Skipping.")
            continue
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
    if task_counter >= end_task:
        print("Finished all assigned tasks.")
        break
    
    
model = LSTMNetKAN(
    input_dim=(len(TVARs)+len(CVARs)-1)*(window_size**2)+1,  # minus 1 because giems2 is target
    hidden_dim=hidden_dim,
    output_dim=1,
    n_layers=n_layers,
    device=device,
    ) 

for lat_idx, lon_idx in zip(lat_idxs, lon_idxs):
    dataset = WetlandDataset(
        lat_idx=lat_idx,
        lon_idx=lon_idx,
        TVARs=TVARs.copy(),
        CVARs=CVARs,
        seq_length=seq_length,
        window_size=window_size,
        start_date=start_date,
        end_date=end_date,
    )
    train_loader, test_loader = wetland_dataloader(
        dataset=dataset,
        train_years=train_years,
        batch_size=batch_size,
    )       
    evaluator = Eval(
        model=model,
        lat_idx=lat_idx,
        lon_idx=lon_idx,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        eval_folder=eval_folder,
        model_folder=model_folder,
        target_scaler=dataset.target_scaler,
        device=device,
    )
    evaluator.run()
    