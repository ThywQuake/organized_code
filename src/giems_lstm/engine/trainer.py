import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import time
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Literal, Any, List
import logging

from ..model import LSTMNet, GRUNet, LSTMNetKAN, GRUNetKAN

# Configure basic logging (can be customized further outside this class)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Add a handler if running standalone without external configuration
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)


class Trainer:
    """
    Handles the entire training process for the sequence models (GRU/LSTM/KAN variants).
    Implements model setup, learning setup, training loop, validation, and early stopping.
    """

    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        learn_rate: float,
        hidden_dim: int,
        n_layers: int,
        n_epochs: int,
        model_type: Literal["GRU", "LSTM", "GRU_KAN", "LSTM_KAN"],
        verbose_epoch: int,
        device: torch.device,
        patience: int,
        debug: bool = False,
    ):
        """
        Initializes the trainer with data loaders, model parameters, and training settings.
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.learn_rate = learn_rate
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.model_type = model_type
        self.verbose_epoch = verbose_epoch
        self.patience = patience
        self.device = device
        self.debug = debug

        if self.debug:
            self.n_epochs = 10
            logger.setLevel(logging.DEBUG)

        # Initialize internal state variables
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.best_loss = float("inf")
        self.epochs_no_improve = 0
        self.epoch_times: List[float] = []
        self.best_model_state = None

    def run(self) -> Any:
        """
        Main entry point for the training process. Runs epochs and loads the best model.
        """
        self._model_setup()
        self._learning_setup()

        logger.info(f"Starting training for {self.n_epochs} epochs on {self.device}")

        for epoch in range(1, self.n_epochs + 1):
            start_time = time.process_time()
            logger.info(f"--- Epoch {epoch}/{self.n_epochs}: ---")

            # 1. Training Phase
            train_avg_loss = self._run_epoch(self.train_loader, is_training=True)

            # 2. Scheduler Step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            logger.info(f"[Train] Avg Loss: {train_avg_loss:.6f}, LR: {current_lr:.6f}")

            # 3. Validation Phase
            test_avg_loss = self._run_epoch(self.test_loader, is_training=False)

            logger.info(f"[Validation] Avg Loss: {test_avg_loss:.6f}")

            # 4. Checkpoint and Early Stopping
            epoch_time = time.process_time() - start_time
            self.epoch_times.append(epoch_time)

            if not self._check_early_stopping(test_avg_loss):
                break

        logger.info(f"Total training time: {sum(self.epoch_times):.2f} seconds")

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Best model loaded with validation loss: {self.best_loss:.6f}")

        return self.model

    def _model_setup(self):
        """
        Initializes the model and moves it to the device.
        """
        input_dim = self.train_loader.dataset[0][0].shape[-1]
        output_dim = 1

        match self.model_type:
            case "LSTM":
                model = LSTMNet(
                    input_dim=input_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=output_dim,
                    n_layers=self.n_layers,
                    device=self.device,
                )
            case "GRU":
                model = GRUNet(
                    input_dim=input_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=output_dim,
                    n_layers=self.n_layers,
                    device=self.device,
                )
            case "LSTM_KAN":
                model = LSTMNetKAN(
                    input_dim=input_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=output_dim,
                    n_layers=self.n_layers,
                    device=self.device,
                )
            case "GRU_KAN":
                model = GRUNetKAN(
                    input_dim=input_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=output_dim,
                    n_layers=self.n_layers,
                    device=self.device,
                )
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        model.to(self.device)
        self.model = model

    def _learning_setup(self):
        """
        Initializes loss criterion (MSELoss), Adam optimizer, and CosineAnnealingLR scheduler.
        """
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learn_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.n_epochs)

    def _init_hidden_state(self, batch_size: int) -> Any:
        """
        Initializes the recurrent hidden state(s) with zeros and moves them to the device.
        """
        h = self.model.init_hidden(batch_size)

        # Ensure the hidden state(s) are on the correct device
        if self.model_type in ["LSTM", "LSTM_KAN"]:
            # LSTM/LSTM-KAN returns a tuple of (h_0, c_0)
            h = tuple([each.data.to(self.device) for each in h])
        else:
            # GRU/GRU-KAN returns h_0
            h = h.data.to(self.device)
        return h

    def _run_batch_step(
        self, inputs: torch.Tensor, targets: torch.Tensor, is_training: bool
    ) -> float:
        """
        Processes a single batch: forward pass, loss calculation, and backpropagation if training.
        """
        inputs = inputs.to(self.device).float()
        targets = targets.to(self.device).float()

        # Initialize hidden state for the batch
        h = self._init_hidden_state(inputs.size(0))

        if is_training:
            self.model.zero_grad()

        # Enable/Disable gradient calculations
        with torch.set_grad_enabled(is_training):
            outputs, h_out = self.model(inputs, h)

            # CRITICAL FIX: Ensure outputs [N, 1] and targets [N, 1] match dimensions.
            loss = self.criterion(outputs, targets)

        if is_training:
            # Standard backward pass
            loss.backward()
            # Gradient clipping
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        return loss.item()

    def _run_epoch(self, data_loader: DataLoader, is_training: bool) -> float:
        """
        Runs one full epoch over the given data loader for training or validation.
        """
        mode = "Training" if is_training else "Validation"
        self.model.train() if is_training else self.model.eval()

        total_loss = 0.0
        counter = 0

        for i, (inputs, targets) in enumerate(data_loader):
            counter += 1

            loss = self._run_batch_step(inputs, targets, is_training)
            total_loss += loss

            if is_training and counter % self.verbose_epoch == 0:
                logger.debug(f"    [{mode}] Batch {counter}, Loss: {loss:.6f}")

        if counter == 0:
            logger.warning(f"Data loader for {mode} is empty.")
            return 0.0

        return total_loss / counter

    def _check_early_stopping(self, current_loss: float) -> bool:
        """
        Checks for early stopping and updates the best model state if improvement is found.
        Returns True to continue training, False to stop.
        """
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.epochs_no_improve = 0
            # Save a deep copy of the model state dict for the best checkpoint
            self.best_model_state = self.model.state_dict().copy()
            logger.info("  New best model state saved.")
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            logger.warning("Early stopping triggered due to no improvement.")
            return False

        return True
