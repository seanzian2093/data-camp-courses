"""
Defines a PyTorch neural network model with sequential layers and computes the output for a given input tensor.
"""

import torch
import torch.nn as nn

# input_tensor shape (1, 11), i.e., input dimension = 11
input_tensor = torch.Tensor([[3, 4, 6, 7, 10, 12, 2, 3, 6, 8, 9]])

# first layer must accept 11-dim
model = nn.Sequential(
    nn.Linear(11, 20),
    # previous layer's output dim = current layer's input dim
    nn.Linear(20, 12),
    nn.Linear(12, 6),
    nn.Linear(6, 4),
)

output = model(input_tensor)
print(output)
