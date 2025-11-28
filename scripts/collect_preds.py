import typer
import sys
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to path so we can import the config module
# Assuming this script is at scripts/collect_preds.py, root is two levels up
sys.path.append(str(Path(__file__).resolve().parents[1]))

from giems_lstm.config import Config

app = typer.Typer(help="Tools for collecting distributed predictions into NetCDF")


def parse_filename(filename: Path):
    """
    Extract lat_idx, lon_idx from filename 'lat_lon.npy'.
    Returns (lat_idx, lon_idx) or None if format is invalid.
    """
    try:
        stem = filename.stem
        parts = stem.split("_")
        if len(parts) != 2:
            return None
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def load_prediction(file_path: Path, lat_idx: int, lon_idx: int):
    """
    Helper function to load a single .npy file.
    Returns tuple (lat_idx, lon_idx, data_array).
    """
    try:
        data = np.load(file_path)
        # Squeeze dimensions if output is (Time, 1) -> (Time,)
        if data.ndim > 1:
            data = data.squeeze()
        return lat_idx, lon_idx, data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


@app.command()
def collect(
    config_path: str = typer.Option(
        "config/E.toml", "--config", "-c", help="Path to config file"
    ),
    output: str = typer.Option(
        "output.nc", "--output", "-o", help="Output NetCDF filename"
    ),
    parallel: int = typer.Option(
        8, "--parallel", "-p", help="Number of threads for reading files"
    ),
    freq: str = typer.Option(
        "MS",
        "--freq",
        "-f",
        help="Pandas frequency string for time axis (e.g., 'MS' for Month Start, 'D' for Day)",
    ),
):
    """
    Collect distributed .npy predictions from the prediction folder into a single NetCDF file.
    Uses parallel I/O and pre-allocation for performance.
    """
    # 1. Load Configuration
    config = Config(config_path, mode="predict")
    pred_folder = config.pred_folder

    if not pred_folder.exists():
        print(f"Error: Prediction folder {pred_folder} does not exist.")
        raise typer.Exit(code=1)

    print(f"Scanning files in {pred_folder}...")
    files = list(pred_folder.glob("*.npy"))
    if not files:
        print("No .npy files found.")
        raise typer.Exit(code=0)

    # 2. Determine Coordinates and Dimensions
    print("Loading reference coordinates from Config...")
    if not config.TVARs:
        print(
            "Error: No TVARs found in config. Ensure config is valid and files exist."
        )
        raise typer.Exit(code=1)

    # We use the first available variable in TVARs to extract the reference Lat/Lon coordinates
    # This ensures the output NetCDF matches the input grid exactly.
    ref_var_name = list(config.TVARs.keys())[0]
    ref_da = config.TVARs[ref_var_name]

    lats = ref_da.coords["lat"].values
    lons = ref_da.coords["lon"].values

    n_lat = len(lats)
    n_lon = len(lons)

    # Generate Time Axis based on Config
    dates = pd.date_range(
        start=config.predict.start_date, end=config.predict.end_date, freq=freq
    )
    n_time = len(dates)

    print(f"Target Grid Shape: Time={n_time}, Lat={n_lat}, Lon={n_lon}")

    # 3. Pre-allocate Global Array
    # Initialize with NaN so missing predictions remain empty
    print("Allocating memory for full dataset...")
    full_array = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

    # 4. Parallel Loading
    print(f"Loading {len(files)} files using {parallel} threads...")

    # Pre-calculate tasks to avoid queuing invalid files
    tasks = []
    for f in files:
        indices = parse_filename(f)
        if indices:
            tasks.append((f, indices[0], indices[1]))

    # Use ThreadPoolExecutor because this is an I/O bound operation
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        # Submit all tasks
        future_to_coords = {
            executor.submit(load_prediction, f, lat, lon): (lat, lon)
            for f, lat, lon in tasks
        }

        # Process results as they complete with a progress bar
        for future in tqdm(
            as_completed(future_to_coords), total=len(tasks), unit="file"
        ):
            result = future.result()
            if result is None:
                continue

            lat_idx, lon_idx, data = result

            # Validation: Ensure data length matches time dimension
            if len(data) == n_time:
                full_array[:, lat_idx, lon_idx] = data
            else:
                # Handle cases where prediction length differs (e.g., truncated)
                valid_len = min(len(data), n_time)
                full_array[:valid_len, lat_idx, lon_idx] = data[:valid_len]

    # 5. Create Xarray Dataset
    print("wrapping data into Xarray Dataset...")
    ds = xr.Dataset(
        data_vars={"prediction": (("time", "lat", "lon"), full_array)},
        coords={"time": dates, "lat": lats, "lon": lons},
        attrs={
            "description": "Aggregated GIEMS-LSTM Predictions",
            "source_script": "collect_preds.py",
            "model_hidden_dim": config.model.hidden_dim,
            "model_type": config.model.type,
        },
    )

    # 6. Save to NetCDF
    output_path = Path(output)
    print(f"Saving to {output_path}...")

    # Use compression to reduce file size (zlib=True)
    encoding = {
        "prediction": {
            "zlib": True,
            "complevel": 5,
            "dtype": "float32",
            "_FillValue": np.nan,
        }
    }
    ds.to_netcdf(output_path, encoding=encoding)

    print(f"Done! Saved {output_path}")


if __name__ == "__main__":
    app()
