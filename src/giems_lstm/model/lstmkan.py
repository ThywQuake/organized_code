import torch
import torch.nn as nn
from .kanlinear import KANLinear


class LSTMNetKAN(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.2
    ):
        super(LSTMNetKAN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob
        ).to(device)
        self.fc = KANLinear(hidden_dim, output_dim).to(device)
        self.relu = nn.ReLU().to(device)

        self.device = device

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:, -1]))
        out = torch.sigmoid(out)
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        # Initialze h_0, c_0 with zeros
        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim)
            .zero_()
            .to(self.device),  # h_0
            weight.new(self.n_layers, batch_size, self.hidden_dim)
            .zero_()
            .to(self.device),
        )
        return hidden
