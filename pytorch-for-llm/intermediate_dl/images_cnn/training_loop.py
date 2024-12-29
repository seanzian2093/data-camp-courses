import torch
import torch.nn as nn
from image_dataset import dataloader
from build_cnn import Net

# Define the model
net = Net(num_classes=7)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Iterate through the epoch
for epoch in range(100):
    running_loss = 0.0
    # Iterate through the DataLoader
    for images, labels in dataloader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(images)

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
