""" In PyTorch, training loop is set up manually."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the model
model = nn.Sequential(
    nn.Linear(11, 20),
    nn.ReLU(),
    nn.Linear(20, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 1),  # Output layer for regression
)

# Example data for regression
data = torch.randn(100, 11)  # 100 samples, 11 features each
labels = torch.randn(100, 1)  # 100 labels (continuous values)

# Create a TensorDataset
dataset = TensorDataset(data, labels)

# Create a DataLoader
data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

# Number of epochs
num_epochs = 5

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Loss function for regression
criterion = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    for batch_data, batch_labels in data_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_data)

        # Compute the loss
        loss = criterion(outputs, batch_labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
