import toml
import xarray as xr
from pathlib import Path
from typing import Literal, Dict, List, Optional
from dataclasses import dataclass


# Define configuration structures for better code completion and type checking
@dataclass
class ModelConfig:
    hidden_dim: int
    n_layers: int
    batch_size: int
    seq_length: int
    window_size: int
    type: str


@dataclass
class TrainConfig:
    start_date: str
    end_date: str
    train_years: List[int]
    lr: float
    n_epochs: int
    patience: int
    verbose_epoch: int


@dataclass
class PredictConfig:
    start_date: str
    end_date: str
    output_file: str
    train_start_date: Optional[str] = None
    train_end_date: Optional[str] = None


@dataclass
class SysConfig:
    debug: bool
    cover_exist: bool
    tasks_per_thread: int


class Config:
    def __init__(
        self, config_path: str, mode: Literal["train", "predict", "analyze"] = "train"
    ):
        self.config_path = Path(config_path)
        # Assuming config is located at config/E.toml, moving up two levels gets to the project root
        self.project_root = self.config_path.parent.parent
        self.mode = mode

        # Load TOML file
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self._raw_config = toml.load(self.config_path)

        # 1. Load parameter configurations (immediate loading)
        self._load_params()

        # 2. Initialize data cache (for lazy loading)
        self._tvars: Optional[Dict[str, xr.DataArray]] = None
        self._cvars: Optional[Dict[str, xr.DataArray]] = None
        self._mask: Optional[xr.DataArray] = None
        # Store raw dataset config for lazy loading usage
        self._dataset_cfg = self._raw_config["dataset"]

    def _load_params(self):
        """Parse normal parameters (non-data related)."""
        # Model
        self.model = ModelConfig(**self._raw_config["model"])

        # Folders (convert to Path objects)
        folder_cfg = self._raw_config["folder"]
        self.model_folder = Path(folder_cfg["model"])
        self.eval_folder = Path(folder_cfg["eval"])
        self.pred_folder = Path(folder_cfg["pred"])

        # Train / Predict / Analyze configs
        if self.mode in ["train", "analyze"]:
            self.train = TrainConfig(**self._raw_config["train"])

        if self.mode in ["predict", "analyze"]:
            # In predict mode, we usually need to know the training time range for scaler consistency
            pred_cfg = self._raw_config["predict"]
            train_cfg = self._raw_config.get("train", {})
            self.predict = PredictConfig(
                start_date=pred_cfg["start_date"],
                end_date=pred_cfg["end_date"],
                output_file=pred_cfg["output_file"],
                train_start_date=train_cfg.get("start_date"),
                train_end_date=train_cfg.get("end_date"),
            )

        # Sys Mode
        sys_cfg = self._raw_config["sys_mode"]
        match self.mode:
            case "train":
                tasks = sys_cfg["train_tasks_per_thread"]
            case "predict":
                tasks = sys_cfg["predict_tasks_per_thread"]
            case "analyze":
                tasks = sys_cfg["collect_tasks_per_thread"]
            case _:
                raise ValueError(f"Unknown mode: {self.mode}")
        self.sys = SysConfig(
            debug=sys_cfg["debug"],
            cover_exist=sys_cfg["cover_exist"],
            tasks_per_thread=tasks,
        )

    # ==================================================
    # Lazy Loading Properties
    # File reading is triggered only when properties (e.g., config.TVARs) are accessed
    # ==================================================

    @property
    def TVARs(self) -> Dict[str, xr.DataArray]:
        if self._tvars is None:
            self._tvars = {}
            for t in self._dataset_cfg["TVARs"]:
                # Use resolve_path to handle relative paths safely
                file_path = self._resolve_path(t["path"])
                # xr.open_dataset is lazy by default, but opening many files still takes time
                ds = xr.open_dataset(file_path)
                self._tvars[t["name"]] = ds[t["variable"]]
        return self._tvars

    @property
    def CVARs(self) -> Dict[str, xr.DataArray]:
        if self._cvars is None:
            self._cvars = {}
            for c in self._dataset_cfg["CVARs"]:
                file_path = self._resolve_path(c["path"])
                ds = xr.open_dataset(file_path)
                self._cvars[c["name"]] = ds[c["variable"]]
        return self._cvars

    @property
    def mask(self):
        if self._mask is None:
            mask_cfg = self._dataset_cfg["mask"]
            file_path = self._resolve_path(mask_cfg["path"])
            self._mask = xr.open_dataset(file_path)[mask_cfg["variable"]].values
            self.total_tasks = mask_cfg["total"]
        return self._mask

    def _resolve_path(self, path_str: str) -> Path:
        """Helper function: Handle paths relative to the project root."""
        path = Path(path_str)
        if path.is_absolute():
            return path
        # If the path in TOML is relative to the project root, we can return it directly
        # or resolve it against self.project_root if strict checking is needed.
        return path


if __name__ == "__main__":
    # Test code
    try:
        config = Config("config/E.toml", mode="predict")
        print(f"Loaded config for model with hidden_dim: {config.model.hidden_dim}")
        print(f"Predict Start Date: {config.predict.start_date}")
        # The following line triggers actual file reading
        # print(config.TVARs.keys())
    except Exception as e:
        print(f"Error loading config: {e}")
