"""
GRU, Gated Recurrent Unit, is a variant of RNN that is designed to solve the vanishing gradient problem of RNN.
It requires less computtion than LSTM and often matches performance, and thus is more efficient.
"""

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the GRU layer
        self.gru = nn.GRU(
            # since we have only one feature
            input_size=1,
            hidden_size=32,
            num_layers=2,
            # batch_first=True means that bach_size will be the first dimention of the RNN input tensor
            batch_first=True,
        )

        # Define the output layer - maps the output of the RNN layer to the target
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # Initialize the hidden state with zeros
        h0 = torch.zeros(2, x.size(0), 32)

        # Forward pass through the GRU layer
        out, _ = self.gru(x, h0)

        # Pass rnn's last output to the output layer - use `-1` in second dimention
        out = self.fc(out[:, -1, :])
        return out
