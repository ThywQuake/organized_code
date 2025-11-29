import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
import numpy as np
from src.giems_lstm.main import app

runner = CliRunner()


@pytest.fixture
def mock_config():
    """
    Mocks the Config class to return controlled values for mask and sys settings.
    """
    with patch("giems_lstm.config.Config") as MockConfig:
        config_instance = MockConfig.return_value

        # Setup mask for calculating chunks (total tasks)
        # Let's say we have 10 tasks total
        mock_mask = MagicMock()
        mock_mask.sum.return_value = 10
        # For numpy.sum(mask) usage
        np_sum_patch = patch("numpy.sum", return_value=10)

        config_instance.mask = np.ones((10, 1))  # Dummy mask

        # Setup sys config
        config_instance.sys.tasks_per_thread = 2

        # Apply the numpy patch as well since main.py might use np.sum(cfg.mask)
        with np_sum_patch:
            yield MockConfig


@patch("src.giems_lstm.main.train_chunk")
def test_train_single_process(mock_train_chunk, mock_config):
    """
    Test 'train' command in single-process mode (default).
    Should call train_chunk directly.
    """
    result = runner.invoke(
        app, ["train", "--config", "dummy.toml", "--thread-id", "5", "--seed", "123"]
    )

    assert result.exit_code == 0
    # Verify train_chunk was called with parsed arguments
    mock_train_chunk.assert_called_once_with(5, "dummy.toml", False, 123)


@patch("src.giems_lstm.main.train_chunk")
@patch("src.giems_lstm.main.mp.Pool")
def test_train_parallel(mock_pool, mock_train_chunk, mock_config):
    """
    Test 'train' command in parallel mode.
    Should initialize Config to calc chunks and then use multiprocessing Pool.
    """
    # Setup mock pool context manager
    pool_instance = mock_pool.return_value.__enter__.return_value

    # Run command with --parallel 4
    result = runner.invoke(app, ["train", "--parallel", "4", "--config", "dummy.toml"])

    assert result.exit_code == 0

    # 1. Check Config was initialized to calculate chunks
    mock_config.assert_called_with("dummy.toml", mode="train")

    # 2. Check Pool was created with correct processes
    mock_pool.assert_called_with(processes=4)

    # 3. Check pool.map was called
    # Total tasks = 10 (from fixture), tasks_per_thread = 2
    # Num chunks = 10 / 2 = 5. range(5) should be passed to map.
    assert pool_instance.map.call_count == 1
    args, _ = pool_instance.map.call_args
    # func is first arg, iterable is second
    assert list(args[1]) == list(range(5))


@patch("src.giems_lstm.main.predict_chunk")
def test_predict_single_process(mock_predict_chunk, mock_config):
    """
    Test 'predict' command in single-process mode.
    """
    result = runner.invoke(app, ["predict", "--thread-id", "2", "--debug"])

    assert result.exit_code == 0
    # Check if debug flag was passed correctly (True) and default seed (3407)
    mock_predict_chunk.assert_called_once_with(2, "config/E.toml", True, 3407)


@patch("src.giems_lstm.main.predict_chunk")
@patch("src.giems_lstm.main.mp.Pool")
def test_predict_parallel(mock_pool, mock_predict_chunk, mock_config):
    """
    Test 'predict' command in parallel mode.
    """
    pool_instance = mock_pool.return_value.__enter__.return_value

    result = runner.invoke(app, ["predict", "--parallel", "2"])

    assert result.exit_code == 0

    mock_config.assert_called_with("config/E.toml", mode="predict")
    mock_pool.assert_called_with(processes=2)
    assert pool_instance.map.call_count == 1


def test_help_command():
    """
    Test that --help works and shows descriptions.
    """
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "GIEMS-LSTM Training and Prediction CLI" in result.stdout
    assert "train" in result.stdout
    assert "predict" in result.stdout
    assert "collect" in result.stdout


# ==============================================================================
# Collect Command Tests
# ==============================================================================


@pytest.fixture
def mock_collect_setup(tmp_path):
    """
    Creates a mock setup with temporary folders for collect tests.
    """
    pred_folder = tmp_path / "pred"
    eval_folder = tmp_path / "eval"
    pred_folder.mkdir()
    eval_folder.mkdir()

    # Create a mock config instance
    mock_config_instance = MagicMock()

    # Small 3x3 mask with some True values
    mock_config_instance.mask = np.array([[True, False, True], [False, True, False], [True, False, True]])

    mock_config_instance.pred_folder = pred_folder
    mock_config_instance.eval_folder = eval_folder

    return mock_config_instance, pred_folder, eval_folder


@patch("giems_lstm.config.Config")
def test_collect_predictions(MockConfig, mock_collect_setup):
    """
    Test collecting prediction results from individual files.
    """
    mock_config_instance, pred_folder, eval_folder = mock_collect_setup
    MockConfig.return_value = mock_config_instance

    # Create some mock prediction files based on the mask
    # Mask has True at (0,0), (0,2), (1,1), (2,0), (2,2)
    coords = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]
    for lat_idx, lon_idx in coords:
        np.save(
            pred_folder / f"{lat_idx}_{lon_idx}.npy",
            {"prediction": np.array([1.0, 2.0, 3.0]), "date": ["2020-01", "2020-02", "2020-03"]},
        )

    output_file = pred_folder / "test_collected.npy"
    result = runner.invoke(
        app,
        ["collect", "--config", "dummy.toml", "--type", "predictions", "--output", str(output_file)],
    )

    assert result.exit_code == 0
    assert "Found 5 result files" in result.stdout
    assert output_file.exists()

    # Verify the collected data
    collected = np.load(output_file, allow_pickle=True).item()
    assert len(collected) == 5
    assert (0, 0) in collected
    assert (2, 2) in collected


@patch("giems_lstm.config.Config")
def test_collect_evaluations(MockConfig, mock_collect_setup):
    """
    Test collecting evaluation results from individual files.
    """
    mock_config_instance, pred_folder, eval_folder = mock_collect_setup
    MockConfig.return_value = mock_config_instance

    # Create some mock evaluation files
    coords = [(0, 0), (0, 2), (1, 1)]
    for lat_idx, lon_idx in coords:
        np.save(
            eval_folder / f"{lat_idx}_{lon_idx}.npy",
            {
                "Y_inv": {"y_pred_train": np.array([1.0]), "y_true_train": np.array([1.0])},
                "metrics": {"train": {"R2": 0.95, "RMSE": 0.1}, "test": {"R2": 0.90, "RMSE": 0.15}},
            },
        )

    output_file = eval_folder / "test_collected.npy"
    result = runner.invoke(
        app,
        ["collect", "--config", "dummy.toml", "--type", "evaluations", "--output", str(output_file)],
    )

    assert result.exit_code == 0
    assert "Found 3 result files" in result.stdout
    assert "=== Evaluation Summary ===" in result.stdout
    assert output_file.exists()


@patch("giems_lstm.config.Config")
def test_collect_missing_files(MockConfig, mock_collect_setup):
    """
    Test collecting when some files are missing.
    """
    mock_config_instance, pred_folder, eval_folder = mock_collect_setup
    MockConfig.return_value = mock_config_instance

    # Only create 2 of the expected 5 files
    np.save(pred_folder / "0_0.npy", {"prediction": np.array([1.0])})
    np.save(pred_folder / "1_1.npy", {"prediction": np.array([2.0])})

    result = runner.invoke(
        app,
        ["collect", "--config", "dummy.toml", "--type", "predictions"],
    )

    assert result.exit_code == 0
    assert "Found 2 result files, 3 missing" in result.stdout


@patch("giems_lstm.config.Config")
def test_collect_no_files(MockConfig, mock_collect_setup):
    """
    Test collecting when no files exist.
    """
    mock_config_instance, pred_folder, eval_folder = mock_collect_setup
    MockConfig.return_value = mock_config_instance

    result = runner.invoke(
        app,
        ["collect", "--config", "dummy.toml", "--type", "predictions"],
    )

    assert result.exit_code == 1
    assert "No result files found" in result.stdout


@patch("giems_lstm.config.Config")
def test_collect_invalid_type(MockConfig, mock_collect_setup):
    """
    Test collecting with invalid result type.
    """
    mock_config_instance, pred_folder, eval_folder = mock_collect_setup
    MockConfig.return_value = mock_config_instance

    result = runner.invoke(
        app,
        ["collect", "--config", "dummy.toml", "--type", "invalid"],
    )

    assert result.exit_code == 1
    assert "Unknown result type" in result.stdout
