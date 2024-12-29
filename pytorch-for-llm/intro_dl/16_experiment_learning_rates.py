""" Example of experimenting with different learning rates in PyTorch """

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generate synthetic data
torch.manual_seed(0)
X = torch.linspace(0, 1, 100).unsqueeze(1)
y = 3 * X + 2 + 0.2 * torch.randn(X.size())


# Define a simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# Function to train the model
def train_model(learning_rate):
    model = LinearRegressionModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    num_epochs = 100
    losses = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


# Experiment with different learning rates
learning_rates = [0.001, 0.01, 0.1, 0.5]
losses_dict = {}

for lr in learning_rates:
    losses_dict[lr] = train_model(lr)

# Plot the losses
plt.figure(figsize=(10, 6))
for lr, losses in losses_dict.items():
    plt.plot(losses, label=f"LR={lr}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Learning Rate Experimentation")
plt.show()
