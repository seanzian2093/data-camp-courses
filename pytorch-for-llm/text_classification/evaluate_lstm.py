import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score
from build_lstm import model, X_train_seq, y_train_seq

# Initialize the metrics
accuracy = Accuracy(task="multiclass", num_classes=3)
precision = Precision(task="multiclass", num_classes=3)
recall = Recall(task="multiclass", num_classes=3)
f1 = F1Score(task="multiclass", num_classes=3)

# Generate predictions
outputs = model(X_train_seq)
_, predicted = torch.max(outputs, 1)

# Update the metrics
accuracy_score = accuracy(predicted, y_train_seq)
precision_score = precision(predicted, y_train_seq)
recall_score = recall(predicted, y_train_seq)
f1_score = f1(predicted, y_train_seq)
print("LSTM Model Evaluation Metrics:")
print(f"Accuracy: {accuracy_score.item():.4f}")
print(f"Precision: {precision_score.item():.4f}")
print(f"Recall: {recall_score.item():.4f}")
print(f"F1 Score: {f1_score.item():.4f}")
