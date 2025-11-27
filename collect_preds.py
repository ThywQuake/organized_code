import numpy as np
import xarray as xr
import os
import pandas as pd

'''============================= Configs ======================================="""'''
mask = xr.open_dataset("data/wetland_mask.nc")
start_date = "1984-01-01"
end_date = "2023-12-01"
pred_folder = "output/pred/E/"
out_file = "output/pred_E.nc"


dates = pd.date_range(start=start_date, end=end_date, freq="MS")
zeros = np.full((len(dates), mask["mask"].shape[0], mask["mask"].shape[1], ), np.nan)
for lat_idx in range(mask["mask"].shape[0]):
    for lon_idx in range(mask["mask"].shape[1]):
        if not mask["mask"].values[lat_idx, lon_idx]:
            continue
        npy_path = f"{pred_folder}{lat_idx}_{lon_idx}.npy"
        if not os.path.exists(npy_path):
            print(f"File {npy_path} does not exist. Skipping...")
            continue
        data = np.load(npy_path, allow_pickle=True).item()
        preds = data["prediction"]
        zeros[lat_idx, lon_idx, :] = preds
        
preds = xr.DataArray(
    data = zeros,
    dims = ["time", "lat", "lon"],
    coords = {        
        "time": dates,
        "lat": mask["lat"].values,
        "lon": mask["lon"].values,

    },
    name = "fwet"
)
preds.to_netcdf(out_file)
