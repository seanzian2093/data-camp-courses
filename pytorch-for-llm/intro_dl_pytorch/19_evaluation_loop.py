""" Example of evaluating a PyTorch model on a dataset. """

import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))

# Generate random data
num_samples = 100
X = torch.randn(num_samples, 10)
y = torch.randn(num_samples, 1)

dataset = torch.utils.data.TensorDataset(X, y)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)

# Initialize criterion and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Evaluation loop
model.eval()
total_loss = 0.0
with torch.no_grad():
    for inputs, targets in data_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

avg_loss = total_loss / len(data_loader)
print(f"Average Loss: {avg_loss:.4f}")
# Since the optimizer is not used in the evaluation loop, it can be removed from the code.

# Turn the model back to training mode
model.train()
