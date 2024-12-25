"""
Creates a simple binary classifier using a neural network with one hidden layer and applies it to an 8-dimensional input tensor.
"""

import torch
import torch.nn as nn

# input_tensor shape (1, 8), i.e., input dimension = 8
input_tensor = torch.Tensor([[3, 4, 6, 2, 3, 6, 8, 9]])

# first layer must accept 8-dim
model = nn.Sequential(nn.Linear(8, 1), nn.Sigmoid())

output = model(input_tensor)
print(output)
