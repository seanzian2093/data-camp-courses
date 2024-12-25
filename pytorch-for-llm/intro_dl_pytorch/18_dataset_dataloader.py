""" Example of using the TensorDataset and DataLoader classes in PyTorch. """

import torch
from torch.utils.data import TensorDataset, DataLoader

# Generate random data
data = torch.randn(100, 10)  # 100 samples, each with 10 features
labels = torch.randint(0, 2, (100,))  # 100 labels (binary classification)

# Create TensorDataset
dataset = TensorDataset(data, labels)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Example of iterating through the DataLoader
for batch_data, batch_labels in dataloader:
    print(batch_data, batch_labels)
    break  # Just to show the first batch
