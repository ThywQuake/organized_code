import os
import sys
import torch
import multiprocessing  # 导入 multiprocessing 库

from utils.model import LSTMNetKAN
from utils.dataset import WetlandDataset
from utils.predict import Predict
from utils.config import Config

import warnings

warnings.filterwarnings("ignore")

"""============================= Configs ======================================="""

print("Loading data...")
config = Config(config_path="config/E_debug.toml", mode="predict")
TVARs = config.TVARs
CVARs = config.CVARs
INPUT_DIM = (len(TVARs) + len(CVARs) - 1) * (config.window_size**2) + 1  # Remove GIEMS-MC and Add potential GIEMS-MC
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

# 提取所有需要的配置参数，以便在多进程中传递
NUM_PROCESSES = min(multiprocessing.cpu_count(), 16)  # 设置进程数，用户可根据硬件调整
HIDDEN_DIM = config.hidden_dim
N_LAYERS = config.n_layers
BATCH_SIZE = config.batch_size
SEQ_LENGTH = config.seq_length
WINDOW_SIZE = config.window_size
START_DATE = config.start_date
END_DATE = config.end_date
TRAIN_START_DATE = config.train_start_date
TRAIN_END_DATE = config.train_end_date

MODEL_FOLDER = config.model_folder
PRED_FOLDER = config.pred_folder
DEBUG_MODE = config.debug
# 注意：Config对象本身包含复杂逻辑，我们尽量避免直接传递

"""============================= Task Function ==================================="""


def process_location(args):
    """
    处理单个地理位置预测任务的函数。
    此函数将在单独的进程中运行。
    """
    (
        lat_idx,
        lon_idx,
        DEBUG_MODE,
        TVARs_copy,
        CVARs_copy,
        device_str,
        HIDDEN_DIM,
        N_LAYERS,
        BATCH_SIZE,
        SEQ_LENGTH,
        WINDOW_SIZE,
        START_DATE,
        END_DATE,
        TRAIN_START_DATE,
        TRAIN_END_DATE,
        MODEL_FOLDER,
        PRED_FOLDER,
        INPUT_DIM,
    ) = args

    # 重新设置设备对象
    local_device = torch.device(device_str)

    print(f"  --> Processing location ({lat_idx}, {lon_idx})")

    os.makedirs(PRED_FOLDER, exist_ok=True)
    save_path = os.path.join(PRED_FOLDER, f"{lat_idx}_{lon_idx}.npy")

    if not DEBUG_MODE and os.path.exists(save_path):
        print(
            f"  --> Predictions for location ({lat_idx}, {lon_idx}) already exist. Skipping..."
        )
        return

    try:
        train_dataset = WetlandDataset(
            TVARs=TVARs_copy.copy(),  # 注意：这里使用传入的副本
            CVARs=CVARs_copy,
            lat_idx=lat_idx,
            lon_idx=lon_idx,
            seq_length=SEQ_LENGTH,
            window_size=WINDOW_SIZE,
            start_date=TRAIN_START_DATE,
            end_date=TRAIN_END_DATE,
            predict=False,
        )
        target_scaler = train_dataset.target_scaler
        feature_scalers = train_dataset.feature_scalers 
        dataset = WetlandDataset(
            TVARs=TVARs_copy.copy(),  # 注意：这里使用传入的副本
            CVARs=CVARs_copy,
            lat_idx=lat_idx,
            lon_idx=lon_idx,
            seq_length=SEQ_LENGTH,
            window_size=WINDOW_SIZE,
            start_date=START_DATE,
            end_date=END_DATE,
            target_scaler=target_scaler,
            feature_scalers=feature_scalers,
            predict=True,
        )

        model = LSTMNetKAN(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=1,
            n_layers=N_LAYERS,
            device=local_device,
        )

        model_path = os.path.join(MODEL_FOLDER, f"{lat_idx}_{lon_idx}.pth")
        if not os.path.exists(model_path):
            print(f"  --> Model file {model_path} does not exist. Skipping...")
            return

        # 注意：这里需要在进程内加载模型
        model.load_state_dict(torch.load(model_path, map_location=local_device))

        pred = Predict(
            dataset=dataset,
            lat_idx=lat_idx,
            lon_idx=lon_idx,
            model=model,
            device=local_device,
            batch_size=BATCH_SIZE,
            save_path=save_path,
        )

        pred.run()
        print(f"  --> Prediction for ({lat_idx}, {lon_idx}) completed.")

    except Exception as e:
        print(f"  --> Error processing ({lat_idx}, {lon_idx}): {e}")


"""============================= Predicting ======================================="""

if __name__ == "__main__":
    print("Starting prediction...")
    task_counter = 0
    task_list = []

    # 遍历所有可能的任务，收集当前线程负责的任务列表
    for lat_idx in range(mask.shape[0]):
        for lon_idx in range(mask.shape[1]):
            if not mask[lat_idx, lon_idx]:
                continue

            if task_counter < start_task:
                task_counter += 1
                continue
            elif task_counter >= end_task:
                # 已经超出当前线程的范围，可以跳出外层循环
                break

            task_counter += 1

            # 将所有必需参数打包成元组，以便传递给进程
            task_args = (
                lat_idx,
                lon_idx,
                DEBUG_MODE,
                TVARs.copy(),
                CVARs.copy(),
                str(device),
                HIDDEN_DIM,
                N_LAYERS,
                BATCH_SIZE,
                SEQ_LENGTH,
                WINDOW_SIZE,
                START_DATE,
                END_DATE,
                TRAIN_START_DATE,
                TRAIN_END_DATE,
                MODEL_FOLDER,
                PRED_FOLDER,
                INPUT_DIM,
            )
            task_list.append(task_args)
        else:
            continue
        break  # 如果内层循环被 break，则外层循环也 break

    print(
        f"Total tasks assigned to this thread (index {temporary_thread}): {len(task_list)}"
    )

    if not task_list:
        print("No tasks to process for this thread.")
    else:
        # 设置并行进程数，通常设置为 CPU 核心数-1 或根据 GPU 数量调整
        # 如果使用 GPU，需要谨慎设置，以防 GPU 内存不足。
        # 建议：如果使用 CPU，设置为 multiprocessing.cpu_count()。
        # 如果使用 GPU，建议设置为 1 或根据实际情况小幅增加。
        # 这里使用 CPU 数量，但用户需要自行根据硬件调整
        print(f"Starting parallel processing with {NUM_PROCESSES} processes...")

        # 创建进程池并运行任务
        with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
            # pool.map 会等待所有任务完成
            pool.map(process_location, task_list)

        print(f"All {len(task_list)} tasks completed for thread {temporary_thread}.")
