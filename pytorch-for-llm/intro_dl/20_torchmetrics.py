""" Example of using the `torchmetrics` library to compute metrics in PyTorch """

import torch
import torchmetrics

# Generate random predictions and targets
# Note: The `task` argument is required for the `Accuracy` metric
preds = torch.randn(10, 5)
targets = torch.randint(0, 5, (10,))

# Initialize the accuracy metric
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=5)

# Compute the accuracy
acc = accuracy(preds, targets)

print(f"Accuracy: {acc.item()}")
# Simulate a batch-wise computation
batch_size = 2
num_batches = preds.size(0) // batch_size

# Initialize the accuracy metric for the epoch
epoch_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=5)

for i in range(num_batches):
    batch_preds = preds[i * batch_size : (i + 1) * batch_size]
    batch_targets = targets[i * batch_size : (i + 1) * batch_size]

    # Compute the accuracy for the batch
    batch_acc = accuracy(batch_preds, batch_targets)
    print(f"Batch {i + 1} Accuracy: {batch_acc.item()}")

    # Update the epoch accuracy
    epoch_accuracy.update(batch_preds, batch_targets)

# Compute the accuracy for the whole epoch
epoch_acc = epoch_accuracy.compute()
print(f"Epoch Accuracy: {epoch_acc.item()}")
