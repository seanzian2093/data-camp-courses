"""Cross entropy is one of the most common ways to measure loss for classification problem"""

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

y = [2]
scores = torch.tensor([[0.1, 6.0, -2.0, 3.2]])

# One hot label
one_hot_label = F.one_hot(torch.tensor(y), num_classes=scores.shape[1])

# Create cross entropy as loss function
criterion = CrossEntropyLoss()

# Calculate loss - must conver to float
loss = criterion(scores.double(), one_hot_label.double())
print(loss)
