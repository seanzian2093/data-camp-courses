import torch
from torchmetrics import MeanSquaredError
from torch.utils.data import DataLoader

from training_loop import net
from sequential_data import dataset_test

# Set up the data loader
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)

# Define the MSE metric
mse = MeanSquaredError()

# Set up the evaluation loop
net.eval()
with torch.no_grad():
    for seqs, labels in dataloader_test:
        seqs = seqs.view(32, 96, 1)
        outputs = net(seqs).squeeze()
        mse(outputs, labels)

# Compute final metric value
test_mse = mse.compute()
print(f"Test MSE: {test_mse}")
