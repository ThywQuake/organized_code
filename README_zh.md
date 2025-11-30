# GIEMS-MC-LSTM

## 概述

**GIEMS-MC-LSTM** 项目实现了一个基于长短期记忆网络（LSTM）的深度学习模型，并在其输出层集成了类似 KAN（Kolmogorov-Arnold Network）的线性层，命名为 `LSTMNetKAN`。

该模型专为大规模地理空间时间序列预测设计，主要用于利用多种环境变量预测湿地动态（如湿地面积分数 `fwet`）。

## 1\. 数据准备 (Data Acquisition)

在运行模型之前，请确保已获取以下必要的源数据。这些数据通常需要处理为 NetCDF 格式，并放置在配置文件（如 `config/E.toml`）指定的路径中。

需要的数据源包括：

  * **GIEMS** (Global Inundation Extent from Multi-Satellites): 提供湿地面积分数 (`fwet`) 作为目标变量。
  * **GRACE** (Gravity Recovery and Climate Experiment): 提供陆地水储量异常 (`lwe_thickness`)。
  * **GLEAM** (Global Land Evaporation Amsterdam Model): 提供土壤湿度 (`sm`)。
  * **MSWEP** (Multi-Source Weighted-Ensemble Precipitation): 提供降水数据 (`pre`)。
  * **ERA5** (ECMWF Reanalysis v5): 提供气温数据 (`tmp`)。

> **注意**：请确保所有数据形状为（time, lat=720, lon=1440）,并且为月尺度分辨率，lat 从 -90° 到 90°，lon 从 -180° 到 180°。

## 2\. 环境安装与配置 (Installation)

本项目使用 **`uv`** 进行现代化的 Python 依赖管理。请按照以下流程进行环境配置：

### 第一步：克隆仓库

```bash
git clone https://github.com/thywquake/GIEMS_MC_LSTM.git
cd GIEMS_MC_LSTM
```

### 第二步：创建虚拟环境

使用 `uv` 创建一个干净的虚拟环境：
(若尚未安装 `uv`，请先前往 [uv 官方文档](https://github.com/astral-sh/uv) 进行安装)

```bash
uv venv
```

*激活虚拟环境（根据您的操作系统）：*

  * Linux/macOS: `source .venv/bin/activate`
  * Windows: `.venv\Scripts\activate`

### 第三步：同步依赖

安装项目所需的依赖包（包括 PyTorch, NumPy 等）：

```bash
uv sync
```

### 第四步：运行测试

在开始训练之前，建议运行测试套件以确保环境和代码正常工作：

```bash
pytest tests/
```

## 3\. 使用方法 (Usage)

安装完成后，您可以使用命令行工具 **`giems`** 来执行训练、预测和结果收集。

### 3.1 训练 (Training)

启动模型训练。

```bash
giems train -c config/E.toml
```

**⚠️ 重要提示：**

  * **不建议**在训练阶段使用 `--parallel` (或 `-p`) 参数。多进程数据加载可能会导致训练不稳定或死锁，建议使用默认的单进程模式或仅指定 `--thread-id` 在集群上分块运行。

| 选项                | 描述                                    | 默认值          |
| :------------------ | :-------------------------------------- | :-------------- |
| `-c`, `--config`    | 配置文件路径                            | `config/E.toml` |
| `-d`, `--debug`     | 调试模式（减少 epoch 数，用于快速测试） | `False`         |
| `-t`, `--thread-id` | 指定运行的任务块 ID（用于集群作业数组） | `0`             |

> **注意**：`--thread-id` 选项需要搭配 `config` 文件中的 `**_tasks_per_thread` 参数使用，以便在集群环境中分块训练、预测、整合。

### 3.2 预测 (Prediction)

使用训练好的模型进行推断。

```bash
# 示例：使用 8 个进程并行预测
giems predict -c config/E.toml -p 8
```

| 选项                | 描述                                    | 默认值          |
| :------------------ | :-------------------------------------- | :-------------- |
| `-c`, `--config`    | 配置文件路径                            | `config/E.toml` |
| `-p`, `--parallel`  | 并行进程数（推荐在预测时使用以加速）    | `1`             |
| `-t`, `--thread-id` | 指定运行的任务块 ID（用于集群作业数组） | `0`             |

### 3.3 结果收集 (Collection)

将分散的预测结果（`.npy` 文件）收集并合并为一个 NetCDF 文件。

```bash
giems collect -c config/E.toml
```

| 选项               | 描述                                                     | 默认值          |
| :----------------- | :------------------------------------------------------- | :-------------- |
| `-c`, `--config`   | 配置文件路径                                             | `config/E.toml` |
| `-p`, `--parallel` | 用于读取文件的并行线程数                                 | `0`             |
| `-e`, `--eval`     | 是否在评估模式下运行(用于收集训练指标，默认收集预测结果) | `False`         |

## 联系方式 (Contact)

如果您在使用过程中遇到任何问题或有合作意向，请联系：

📧 **Email**: [thywquake@foxmail.com](mailto:thywquake@foxmail.com)
