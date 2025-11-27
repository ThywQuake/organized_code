import torch
import torch.nn as nn

from .kanlinear import KANLinear


class GRUNetKAN(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.2
    ):
        super(GRUNetKAN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(
            input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob
        )
        self.fc = KANLinear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

        self.device = device

    def forward(self, x, h):
        out, h = self.gru(x, h)
        # print(out[:, -1].shape, h.shape)
        # select hidden state of last timestamp (t=90) (1024, 256)
        out = self.fc(self.relu(out[:, -1]))  # out[:, -1, :]
        # print(out.shape) # (1024, 1)
        out = torch.sigmoid(out)
        return out, h

    def init_hidden(self, batch_size):
        # Initialze h_0 with zeros
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim)
            .zero_()
            .to(self.device)
        )
        return hidden
