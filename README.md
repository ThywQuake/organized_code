Here is the English transcription of the provided Chinese text.

# GIEMS-MC-LSTM

## Overview

The **GIEMS-MC-LSTM** project implements a deep learning model based on the Long Short-Term Memory network (LSTM), which integrates a KAN (Kolmogorov-Arnold Network)-like linear layer in its output, named `LSTMNetKAN`.

This model is specifically designed for **large-scale geospatial time series forecasting**, primarily used to predict wetland dynamics (such as the wetland fraction area `fwet`) utilizing multiple environmental variables.

## 1\. Data Acquisition

Before running the model, please ensure you have acquired the following necessary source data. This data usually needs to be processed into NetCDF format and placed in the path specified by the configuration file (e.g., `config/E.toml`).

Required data sources include:

  * **GIEMS** (Global Inundation Extent from Multi-Satellites): Provides the wetland fraction area (`fwet`) as the target variable.
  * **GRACE** (Gravity Recovery and Climate Experiment): Provides land water equivalent thickness anomaly (`lwe_thickness`).
  * **GLEAM** (Global Land Evaporation Amsterdam Model): Provides soil moisture (`sm`).
  * **MSWEP** (Multi-Source Weighted-Ensemble Precipitation): Provides precipitation data (`pre`).
  * **ERA5** (ECMWF Reanalysis v5): Provides air temperature data (`tmp`).

> **Note**: Please ensure all data has the shape (time, lat=720, lon=1440) and is at a monthly resolution, with latitude ranging from -90° to 90° and longitude from -180° to 180°.

## 2\. Installation and Configuration

This project uses **`uv`** for modern Python dependency management. Please follow the steps below to configure your environment:

### Step 1: Clone the Repository

```bash
git clone https://github.com/thywquake/GIEMS_MC_LSTM.git
cd GIEMS_MC_LSTM
```

### Step 2: Create a Virtual Environment

Use `uv` to create a clean virtual environment:
(If you have not installed `uv`, please go to the [uv official documentation](https://github.com/astral-sh/uv) for installation first)

```bash
uv venv
```

*Activate the virtual environment (depending on your operating system):*

  * Linux/macOS: `source .venv/bin/activate`
  * Windows: `.venv\Scripts\activate`

### Step 3: Sync Dependencies

Install the project's required dependencies (including PyTorch, NumPy, etc.):

```bash
uv sync
```

### Step 4: Run Tests

Before starting the training, it is recommended to run the test suite to ensure the environment and code are working correctly:

```bash
pytest tests/
```

## 3\. Usage

After installation, you can use the command-line tool **`giems`** to perform training, prediction, and result collection.

### 3.1 Training

Start the model training.

```bash
giems train -c config/E.toml
```

**⚠️ Important Tip:**

  * It is **not recommended** to use the `--parallel` (or `-p`) parameter during the training phase. Multi-process data loading may lead to training instability or deadlocks. It is suggested to use the default single-process mode or only specify `--thread-id` to run in blocks on a cluster.

| Option           | Description                                                 | Default Value   |
| :--------------- | :---------------------------------------------------------- | :-------------- |
| `-c`, `--config` | Path to the configuration file                              | `config/E.toml` |
| `-d`, `--debug`  | Debug mode (reduces the number of epochs for quick testing) | `False`         |

> **Note**: The `--thread-id` option needs to be used in conjunction with the `**_tasks_per_thread` parameter in the `config` file to enable block-wise training, prediction, and integration in a cluster environment.

### 3.2 Prediction

Perform inference using the trained model.

```bash
# Example: Use 8 processes for parallel prediction
giems predict -c config/E.toml -p 8
```

| Option              | Description                                                            | Default Value   |
| :------------------ | :--------------------------------------------------------------------- | :-------------- |
| `-c`, `--config`    | Path to the configuration file                                         | `config/E.toml` |
| `-p`, `--parallel`  | Number of parallel processes (recommended for prediction acceleration) | `1`             |
| `-t`, `--thread-id` | Specifies the task block ID to run (for cluster job arrays)            | `0`             |

### 3.3 Result Collection

Collect scattered prediction results (`.npy` files) and merge them into a single NetCDF file.

```bash
giems collect -c config/E.toml
```

| Option             | Description                                                                                                    | Default Value   |
| :----------------- | :------------------------------------------------------------------------------------------------------------- | :-------------- |
| `-c`, `--config`   | Path to the configuration file                                                                                 | `config/E.toml` |
| `-p`, `--parallel` | Number of parallel threads used for reading files                                                              | `0`             |
| `-e`, `--eval`     | Whether to run in evaluation mode (for collecting training metrics, defaults to collecting prediction results) | `False`         |

## Contact

If you encounter any issues during usage or are interested in collaboration, please contact:

📧 **Email**: [thywquake@foxmail.com](mailto:thywquake@foxmail.com)

-----

Would you like me to clarify any specific section, or perhaps help draft an email based on this contact information?
