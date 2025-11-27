import os
import sys
import numpy as np
import torch

from utils.model import LSTMNetKAN
from utils.dataset import WetlandDataset, wetland_dataloader
from utils.train import Train
from utils.eval import Eval
from utils.config import Config

import warnings

warnings.filterwarnings("ignore")

"""============================= Configs ======================================="""

print("Loading data...")
config = Config(config_path="config/E.toml", mode="train")
TVARs = config.TVARs
CVARs = config.CVARs
mask = config.mask
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

total_tasks = int(np.sum(mask))
print(f"Total training locations: {total_tasks}")
if config.debug:
    temporary_thread = "0"
else:
    temporary_thread = sys.argv[1]

tasks_per_thread = config.tasks_per_thread
start_task = int(temporary_thread) * tasks_per_thread
end_task = min((int(temporary_thread) + 1) * tasks_per_thread, total_tasks)


train_years = config.train_years
lr = config.lr
hidden_dim = config.hidden_dim
n_layers = config.n_layers
n_epochs = config.n_epochs
verbose_epoch = config.verbose_epoch
patience = config.patience
batch_size = config.batch_size
seq_length = config.seq_length
window_size = config.window_size
start_date = config.start_date
end_date = config.end_date

model_folder = config.model_folder
eval_folder = config.eval_folder
"""============================= Training & Evaluation ======================================="""

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
        print(
            f"Processing location {task_counter}/{total_tasks} at lat_idx: {lat_idx}, lon_idx: {lon_idx}"
        )

        os.makedirs(model_folder, exist_ok=True)
        model_path = os.path.join(model_folder, f"{lat_idx}_{lon_idx}.pth")
        if not config.debug and os.path.exists(model_path):
            print(
                f"Model for location at lat_idx: {lat_idx}, lon_idx: {lon_idx} already exists. Skipping training."
            )
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
            model_type="LSTM_KAN",
            verbose_epoch=verbose_epoch,
            device=device,
            patience=patience,
            debug=config.debug,
        )
        model = trainer.run()
        if model is None:
            print(
                f"Training failed for location at lat_idx: {lat_idx}, lon_idx: {lon_idx}. Skipping."
            )
            continue
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    if task_counter >= end_task:
        print("Finished all assigned tasks.")
        break

input_dim = (len(TVARs) + len(CVARs) - 1) * (window_size**2) + 1 # Remove GIEMS-MC and Add potential GIEMS-MC
model = LSTMNetKAN(
    input_dim=input_dim,
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
