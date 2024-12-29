""" Use an optimier to update weight automatically. """

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

input_tensor = torch.Tensor([[3, 4, 6, 7, 10, 12, 2, 3, 6, 8, 9]])

model = nn.Sequential(
    nn.Linear(11, 20),
    nn.Linear(20, 12),
    nn.Linear(12, 6),
    nn.Linear(6, 4),
    nn.Softmax(dim=1),
)

y = [2]
num_classes = 4
# One hot label
target = F.one_hot(torch.tensor(y), num_classes=num_classes)

# Predictions
pred = model(input_tensor)

# Create cross entropy as loss function
criterion = CrossEntropyLoss()

# Create optimizer - using `.parameters()` to access a model's parameters
optimizer = optim.SGD(model.parameters(), lr=0.9)

# Calculate loss
loss = criterion(pred, target.double())
print("Before optimization loss is: ", loss)

# Update model's parameters using optimizer
loss1 = optimizer.step()
print("loss is: ", loss)
