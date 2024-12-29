""" Use Mean Squared Error(MSE) as a loss function for regression"""

import torch
import torch.nn as nn
import numpy as np

# y and y_hat
y_pred = np.array(10)
y = np.array(1)

# MSELoss - numpy
mse_np = (y_pred - y) ** 2

# MSELoss - Pytorch
criterion = nn.MSELoss()

mse_pytorch = criterion(
    # must convert to `.double()`
    torch.tensor(y_pred).double(),
    torch.tensor(y).double(),
)
print(mse_pytorch)
