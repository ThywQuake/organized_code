import os
import sys
import numpy as np
import torch
import multiprocessing
import warnings
from torch.nn.utils import clip_grad_norm_

from utils.model import LSTMNetKAN
from utils.dataset import WetlandDataset, wetland_dataloader
from utils.train import Train
from utils.config import Config

warnings.filterwarnings("ignore")


def train_single_location(args):
    """
    单个位置的训练任务，将在子进程中运行
    """
    (
        lat_idx,
        lon_idx,
        config_dict,  # 传递配置字典而不是对象
        TVARs_copy,
        CVARs_copy,
        device_str,
    ) = args

    # ---------------------------------------------------------
    # 【关键优化】如果是 CPU 训练，强制限制每个进程只用 1 个核
    # 防止 N 个进程 * N 个核 导致的资源争抢灾难
    if device_str == "cpu":
        torch.set_num_threads(1)
    # ---------------------------------------------------------

    device = torch.device(device_str)

    # 解包配置参数
    model_folder = config_dict["model_folder"]
    train_years = config_dict["train_years"]
    debug = config_dict["debug"]

    # 路径检查
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, f"{lat_idx}_{lon_idx}.pth")

    # 跳过已存在的模型
    if not debug and os.path.exists(model_path):
        # print(f"Exists: {lat_idx}, {lon_idx}")
        return

    try:
        # 1. 准备数据
        dataset = WetlandDataset(
            lat_idx=lat_idx,
            lon_idx=lon_idx,
            TVARs=TVARs_copy,  # 这里已经是副本
            CVARs=CVARs_copy,
            seq_length=config_dict["seq_length"],
            window_size=config_dict["window_size"],
            start_date=config_dict["start_date"],
            end_date=config_dict["end_date"],
        )

        train_loader, test_loader = wetland_dataloader(
            dataset=dataset,
            train_years=train_years,
            batch_size=config_dict["batch_size"],
            num_workers=0,  # 子进程中不要再开 dataloader 的 worker
        )

        # 2. 初始化训练器
        trainer = Train(
            train_loader=train_loader,
            test_loader=test_loader,
            learn_rate=config_dict["lr"],
            hidden_dim=config_dict["hidden_dim"],
            n_layers=config_dict["n_layers"],
            n_epochs=config_dict["n_epochs"],
            model_type="LSTM_KAN",
            verbose_epoch=config_dict["verbose_epoch"],
            device=device,
            patience=config_dict["patience"],
            debug=debug,
        )

        # 3. 运行训练
        print(f"--> Training start: ({lat_idx}, {lon_idx}) on {device}")
        model = trainer.run()

        if model is not None:
            torch.save(model.state_dict(), model_path)
            print(f"--> Saved: {model_path}")
        else:
            print(f"--> Failed: ({lat_idx}, {lon_idx})")

    except Exception as e:
        print(f"--> Error at ({lat_idx}, {lon_idx}): {e}")


if __name__ == "__main__":
    print("Loading data for parallel training...")
    # 配置文件加载
    config = Config(config_path="config/E_debug.toml", mode="train")

    # 将 Config 对象转为纯字典，方便序列化传递给子进程
    config_dict = {
        "model_folder": config.model_folder,
        "train_years": config.train_years,
        "debug": config.debug,
        "seq_length": config.seq_length,
        "window_size": config.window_size,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "hidden_dim": config.hidden_dim,
        "n_layers": config.n_layers,
        "n_epochs": config.n_epochs,
        "verbose_epoch": config.verbose_epoch,
        "patience": config.patience,
        "tasks_per_thread": config.tasks_per_thread,
    }

    TVARs = config.TVARs
    CVARs = config.CVARs
    mask = config.mask

    # 【重要】决定设备
    # 如果打算并行训练，强烈建议强制使用 cpu，除非你有 H100/A100 80GB 这种大显存卡
    # device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device_str = "cpu"  # 推荐：并行训练时使用 CPU

    print(f"Target Device for workers: {device_str}")

    # 计算任务分配
    total_tasks = int(np.sum(mask))
    if config.debug:
        temporary_thread = "0"
    else:
        try:
            temporary_thread = sys.argv[1]
        except IndexError:
            temporary_thread = "0"

    start_task = int(temporary_thread) * config.tasks_per_thread
    end_task = min((int(temporary_thread) + 1) * config.tasks_per_thread, total_tasks)

    # 准备任务列表
    task_list = []
    task_counter = 0

    print("Generating task list...")
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

            # 打包参数
            # 注意：TVARs.copy() 对内存消耗很大，如果内存不够，可以考虑共享内存
            # 但 xarray/numpy 在 Linux 下有多进程写时复制优化(COW)，只读读取通常还好
            args = (
                lat_idx,
                lon_idx,
                config_dict,
                TVARs.copy(),
                CVARs.copy(),
                device_str,
            )
            task_list.append(args)

        if task_counter >= end_task:
            break

    print(f"Total tasks to run: {len(task_list)}")

    # 设置并行数
    # 如果是 CPU 训练，可以设为 cpu_count()
    # 如果是 GPU 训练，建议设为 1 或 2，否则显存爆炸
    num_processes = min(multiprocessing.cpu_count(), 16)

    if len(task_list) > 0:
        print(f"Starting Pool with {num_processes} processes...")
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(train_single_location, task_list)

    print("All training tasks completed.")
