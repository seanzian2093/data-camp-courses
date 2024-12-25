"""
This script demonstrates how to create one-hot encoded vectors using both NumPy and PyTorch.
"""

import torch
import numpy as np
import torch.nn.functional as F

y = 1
num_classes = 3

# one-hot encoded vector using numpy - manually encoding
one_hot_np = np.array([0, 1, 0])
print(one_hot_np)

# using PyTorch
one_hot_pytorch = F.one_hot(torch.tensor(y), num_classes=num_classes)
print(one_hot_pytorch)
