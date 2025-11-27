import os
import sys
import torch

from utils.model import LSTMNetKAN
from utils.dataset import WetlandDataset
from utils.pred import Predict
from utils.config import Config

import warnings

warnings.filterwarnings("ignore")

"""============================= Configs ======================================="""

print("Loading data...")
config = Config(config_path="config/E_debug.toml", mode="predict")
TVARs = config.TVARs
CVARs = config.CVARs
mask = config.mask
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

total_tasks = config.total_tasks
print(f"Total training locations: {total_tasks}")
tasks_per_thread = config.tasks_per_thread
if config.debug:
    temporary_thread = "0"
else:
    temporary_thread = sys.argv[1]

start_task = int(temporary_thread) * tasks_per_thread
end_task = min((int(temporary_thread) + 1) * tasks_per_thread, total_tasks)

hidden_dim = config.hidden_dim
n_layers = config.n_layers
batch_size = config.batch_size
seq_length = config.seq_length
window_size = config.window_size
start_date = config.start_date
end_date = config.end_date

model_folder = config.model_folder
eval_folder = config.eval_folder
pred_folder = config.pred_folder
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
        if not config.debug and os.path.exists(save_path):
            print(
                f"Predictions for location ({lat_idx}, {lon_idx}) already exist. Skipping..."
            )
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
            predict=True,
        )
        input_dim = (len(TVARs) + len(CVARs) - 1) * (window_size**2) + 1

        model = LSTMNetKAN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            n_layers=n_layers,
            device=device,
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
            model=model,
            device=device,
            batch_size=batch_size,
            save_path=save_path,
        )

        pred.run()
