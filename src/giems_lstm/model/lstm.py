import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.2
    ):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:, -1]))
        out = torch.sigmoid(out)  # （0，1）
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
