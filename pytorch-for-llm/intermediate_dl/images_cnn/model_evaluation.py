import torch
from torchmetrics import Precision, Recall
from training_loop import net
from image_dataset import dataloader

# Define metrics
# use micro to calculate the metric globally
# use macro to calculate the metric for each class and then average them
metrics_precision = Precision(task="multiclass", num_classes=7, average="micro")
metrics_recall = Recall(task="multiclass", num_classes=7, average="micro")

net.eval()
# Iterate through the DataLoader of test data

with torch.no_grad():
    for images, labels in dataloader:
        # Forward pass
        outputs = net(images)

        # Get the predicted class
        _, predicted = torch.max(outputs, 1)

        # Calculate the precision and recall
        metrics_precision(predicted, labels)
        metrics_recall(predicted, labels)

# Get the precision and recall
precision = metrics_precision.compute()
recall = metrics_recall.compute()
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

# Get the precision and recall for each class
precision_per_class = {
    k: precision[v].item() for k, v in dataloader.dataset.class_to_idx.items()
}
print(f"Precision per class: {precision_per_class}")

recall_per_class = {
    k: recall[v].item() for k, v in dataloader.dataset.class_to_idx.items()
}
print(f"recall per class: {recall_per_class}")
