import torch
from torchmetrics import Accuracy


def evaluate_model(model, dataloader_test):
    # Define accuracy for each output
    acc_alpha = Accuracy(task="multiclass", num_classes=30)
    acc_char = Accuracy(task="multiclass", num_classes=964)

    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        for images, labels_alpha, labels_char in dataloader_test:
            # Generate predictions
            outputs_alpha, outputs_char = model(images)
            _, pred_alpha = torch.max(outputs_alpha, 1)
            _, pred_char = torch.max(outputs_alpha, 1)
            # Update accuracy
            acc_alpha(pred_alpha, labels_alpha)
            acc_char(pred_char, labels_char)

    print(f"Accuracy - Alphabet: {acc_alpha.compute()}")
    print(f"Accuracy - Charater: {acc_char.compute()}")
