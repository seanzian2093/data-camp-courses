import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from build_lstm import Net
from sequential_data import dataset_train

net = Net(input_size=1)
print(net)

# Set up MSE loss
criterion = nn.MSELoss()

# Setup Adam optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

# Set up the data loader
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

# Set up the training loop
for epoch in range(10):
    for seqs, labels in dataloader_train:
        # for seqs, labels in dataset_train:
        # Reshape the mode inputs
        # print(seqs.shape)
        seqs = seqs.view(32, 96, 1)

        # Run through model
        outputs = net(seqs)

        # Compute loss
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
