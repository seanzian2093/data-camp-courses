""" Example of experimenting with different momentum in PyTorch """

import torch
import torch.nn as nn
import torch.optim as optim

# Generate random data
input_size = 28 * 28
num_classes = 10
num_samples = 60000

# Random dataset
inputs = torch.randn(num_samples, input_size)
labels = torch.randint(0, num_classes, (num_samples,))

# Create a DataLoader for the random dataset
random_dataset = torch.utils.data.TensorDataset(inputs, labels)
trainloader = torch.utils.data.DataLoader(random_dataset, batch_size=64, shuffle=True)

# Reinitialize the network, loss function, and optimizer with momentum
net = nn.Sequential(
    nn.Linear(input_size, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, num_classes),
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# Training loop with random data
for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print("Finished Training")
