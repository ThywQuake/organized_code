import torch 
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import time 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Literal

from utils.model import LSTMNet, GRUNet, LSTMNetKAN, GRUNetKAN 


class Train:
    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        learn_rate: float,
        hidden_dim: int,
        n_layers: int,
        n_epochs: int,
        model_type: Literal['GRU', 'LSTM', 'GRU_KAN', 'LSTM_KAN'],
        verbose_epoch: int,
        device: torch.device,
        patience: int,
        debug: bool = False,
    ):
        """
        Initialize the training process with data loaders, model parameters, and training settings.
        
        Args:
            train_loader (DataLoader): DataLoader for training data.
            test_loader (DataLoader): DataLoader for testing data.
            learn_rate (float): Learning rate for the optimizer.
            hidden_dim (int): Dimension of the hidden layers in the model.
            n_layers (int): Number of layers in the model.
            n_epochs (int): Number of epochs to train the model.
            model_type (Literal['GRU', 'LSTM', 'GRU_KAN', 'LSTM_KAN']): Type of model to use.
            verbose_epoch (int): Frequency of logging training progress.
            patience (int): Patience for early stopping.
            device (torch.device): Device to run the training on.
            debug (bool): If True, runs in debug mode with limited data.
        """
        
        self.train_loader, self.test_loader = train_loader, test_loader
        self.learn_rate = learn_rate
        self.hidden_dim, self.n_layers = hidden_dim, n_layers
        self.n_epochs = n_epochs
        self.model_type = model_type
        self.verbose_epoch = verbose_epoch
        self.patience = patience
        self.device = device
        self.debug = debug
        
        if self.debug:
            self.n_epochs = 10
        
    def run(self):
        self.model_setup()
        self.learning_setup()
        
        for epoch in range(1, self.n_epochs + 1):
            print(f"Epoch {epoch}/{self.n_epochs}:")
            continue_training = self.train_one_epoch()
            if not continue_training:
                break
                
        print(f"Total training time: {sum(self.epoch_times):.2f} seconds")
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Best model loaded with loss: {self.best_loss:.6f}")
            
        return self.model
        
    def model_setup(self):
        if self.train_loader is not None and len(self.train_loader.dataset) > 0:
            input_dim = self.train_loader.dataset[0][0][0].shape[0]
        else:
            input_dim = 244  # Default input dimension if train_loader is empty
            
        output_dim = 1
        
        match self.model_type:
            case 'LSTM':
                model = LSTMNet(input_dim, self.hidden_dim, output_dim, self.n_layers)
            case 'GRU':
                model = GRUNet(input_dim, self.hidden_dim, output_dim, self.n_layers)
            case 'LSTM_KAN':
                model = LSTMNetKAN(input_dim, self.hidden_dim, output_dim, self.n_layers, device=self.device)
            case 'GRU_KAN':
                model = GRUNetKAN(input_dim, self.hidden_dim, output_dim, self.n_layers)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
        model.to(self.device)
        self.model = model
        
    def learning_setup(self):
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learn_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.n_epochs)
        
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        self.epoch_times = []
        self.best_model_state = None
        
    def train_one_epoch(self):
        start_time = time.process_time()
        avg_loss = 0.0
        counter = 0
        
        self.model.train()
        for inputs, targets in self.train_loader:
            counter += 1
            loss = self.train_phase(inputs, targets)
            avg_loss += loss.item()
            
            if counter % self.verbose_epoch == 0:
                print(f"    [Train] Batch {counter}, Loss: {loss.item():.6f}")
                
        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"  [Train] Epoch completed. Avg Loss: {avg_loss / counter:.6f}, LR: {current_lr:.6f}")
        
        self.model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for inputs, targets in self.test_loader:
                loss = self.test_phase(inputs, targets)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(self.test_loader)
        print(f"  [Test] Avg Loss: {avg_test_loss:.6f}")
        
        epoch_time = time.process_time() - start_time
        self.epoch_times.append(epoch_time)
        
        if avg_test_loss < self.best_loss:
            self.best_loss = avg_test_loss
            self.epochs_no_improve = 0
            self.best_model_state = self.model.state_dict().copy()
        else:
            self.epochs_no_improve += 1
            
        if self.epochs_no_improve >= self.patience:
            print("Early stopping triggered.")
            return False
        
        return True
    
    def train_phase(self, inputs, targets):
        inputs = inputs.to(self.device).float()
        targets = targets.to(self.device).float()
        
        h = self.model.init_hidden(inputs.size(0))
        if self.model_type in ['LSTM', 'LSTM_KAN']:
            h = tuple([each.data.to(self.device) for each in h])
        else:
            h = h.data 
            
        self.model.zero_grad()
        outputs, h = self.model(inputs, h)
        loss = self.criterion(outputs, targets.squeeze())
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=1.0)  
        self.optimizer.step()
        
        return loss
    
    def test_phase(self, inputs, targets):
        inputs = inputs.to(self.device).float()
        targets = targets.to(self.device).float()
        
        h = self.model.init_hidden(inputs.size(0))
        if self.model_type in ['LSTM', 'LSTM_KAN']:
            h = tuple([each.data.to(self.device) for each in h])
        else:
            h = h.data 
            
        outputs, h = self.model(inputs, h)
        loss = self.criterion(outputs, targets.squeeze())
        
        return loss
        
if __name__ == "__main__":
    ...