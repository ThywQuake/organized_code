import numpy as np


def _allocate_coords(
    mask: np.ndarray, start_task: int, end_task: int, left: bool = False
):
    task_counter = 0
    lats_gen = range(mask.shape[0]) if not left else range(mask.shape[0] - 1, -1, -1)
    lons_gen = range(mask.shape[1]) if not left else range(mask.shape[1] - 1, -1, -1)
    for lat_idx in lats_gen:
        for lon_idx in lons_gen:
            if not mask[lat_idx, lon_idx]:
                continue
            if task_counter < start_task:
                task_counter += 1
                continue
            elif task_counter >= end_task:
                return

            task_counter += 1
            yield (lat_idx, lon_idx)
