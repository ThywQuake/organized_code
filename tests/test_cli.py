import pytest
from typer.testing import CliRunner
from unittest.mock import patch

# Import the main Typer app
# Ensure 'src' is in your PYTHONPATH if running tests locally without installing the package
from giems_lstm.main import app

# Initialize the CLI runner for Typer
runner = CliRunner()


@pytest.fixture
def mock_functions():
    """
    Fixture to mock the core logic functions in main.py.
    This prevents actual training/prediction code from running during CLI testing.
    """
    with (
        patch("giems_lstm.main._train") as mock_train,
        patch("giems_lstm.main._predict") as mock_predict,
        patch("giems_lstm.main._collect") as mock_collect,
        patch("giems_lstm.main._uniform_entry") as mock_entry,
    ):
        yield {
            "train": mock_train,
            "predict": mock_predict,
            "collect": mock_collect,
            "entry": mock_entry,
        }


def test_app_help():
    """
    Test that the --help flag works and displays the correct description and commands.
    """
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "GIEMS-LSTM Training and Prediction CLI" in result.stdout
    assert "train" in result.stdout
    assert "predict" in result.stdout
    assert "collect" in result.stdout


def test_train_command_defaults(mock_functions):
    """
    Test the 'train' command with default arguments.
    """
    result = runner.invoke(app, ["train"])

    assert result.exit_code == 0

    # Verify _uniform_entry is called with defaults (debug=False, parallel=0, seed=3407)
    mock_functions["entry"].assert_called_once_with(False, 0, 3407)

    # Verify _train is called with defaults (thread_id=0, config_path="config/F.toml", debug=False, parallel=0)
    mock_functions["train"].assert_called_once_with(0, "config/F.toml", False, 0)


def test_train_command_custom_args(mock_functions):
    """
    Test the 'train' command with custom arguments.
    """
    args = [
        "train",
        "--config",
        "config/custom.toml",
        "--thread-id",
        "5",
        "--parallel",
        "4",
        "--debug",
        "--seed",
        "12345",
    ]
    result = runner.invoke(app, args)

    assert result.exit_code == 0

    # Verify arguments are correctly passed to the entry point
    mock_functions["entry"].assert_called_once_with(True, 4, 12345)

    # Verify arguments are correctly passed to the train function
    mock_functions["train"].assert_called_once_with(5, "config/custom.toml", True, 4)


def test_predict_command_defaults(mock_functions):
    """
    Test the 'predict' command with default arguments.
    """
    result = runner.invoke(app, ["predict"])

    assert result.exit_code == 0

    mock_functions["entry"].assert_called_once_with(False, 0, 3407)
    # _predict(thread_id, config_path, debug, parallel)
    mock_functions["predict"].assert_called_once_with(0, "config/F.toml", False, 0)


def test_predict_command_custom_args(mock_functions):
    """
    Test the 'predict' command with custom arguments using short flags.
    """
    args = ["predict", "-c", "config/pred.toml", "-t", "2", "-p", "8", "-d", "-s", "42"]
    result = runner.invoke(app, args)

    assert result.exit_code == 0

    mock_functions["entry"].assert_called_once_with(True, 8, 42)
    mock_functions["predict"].assert_called_once_with(2, "config/pred.toml", True, 8)


def test_collect_command_defaults(mock_functions):
    """
    Test the 'collect' command with default arguments.
    """
    result = runner.invoke(app, ["collect"])

    assert result.exit_code == 0

    # collect command has default parallel=0 in CLI definition, but note that
    # some logic might use a default if passed 0. Here we check strictly what CLI passes.
    # Based on the uploaded file, the default parallel for collect is 0 in main.py.
    mock_functions["entry"].assert_called_once_with(False, 0, 3407)

    # _collect(config_path, eval, parallel)
    # The default for eval is False
    mock_functions["collect"].assert_called_once_with("config/F.toml", False, 0)


def test_collect_command_custom_args(mock_functions):
    """
    Test the 'collect' command with custom arguments.
    """
    args = [
        "collect",
        "--config",
        "config/analysis.toml",
        "--parallel",
        "16",
        "--seed",
        "999",
        "--eval",
    ]
    result = runner.invoke(app, args)

    assert result.exit_code == 0

    mock_functions["entry"].assert_called_once_with(False, 16, 999)
    # Check that eval is passed as True
    mock_functions["collect"].assert_called_once_with("config/analysis.toml", True, 16)


def test_invalid_argument():
    """
    Test that providing an invalid argument results in a failure.
    """
    result = runner.invoke(app, ["train", "--invalid-arg", "123"])

    assert result.exit_code != 0
    # Check .output (stdout + stderr) for the error message
    assert "No such option" in result.output


def test_entry_point_logic_with_parallel_check():
    """
    Test the internal logic of _uniform_entry function, specifically how it handles
    logging setup and multiprocessing initialization.
    """
    from giems_lstm.main import _uniform_entry

    with (
        patch("giems_lstm.main._setup_global_logging") as mock_logging,
        patch("giems_lstm.main._seed_everything") as mock_seed,
        patch("multiprocessing.cpu_count", return_value=8),
        patch("torch.set_num_threads") as mock_torch_threads,
    ):
        # Case 1: parallel=0 (Standard single process, no specific mp setup)
        _uniform_entry(debug=False, parallel=0, seed=123)
        mock_logging.assert_called_with(False, "Main")
        mock_seed.assert_called_with(123)
        mock_torch_threads.assert_not_called()

        # Case 2: parallel=4 (Parallel mode enabled)
        # Logic: num_workers = cpu_count // parallel = 8 // 4 = 2
        _uniform_entry(debug=True, parallel=4, seed=456)
        mock_logging.assert_called_with(True, "Main")
        mock_seed.assert_called_with(456)
        # Ensure torch threads are set based on the calculation
        mock_torch_threads.assert_called_with(2)
