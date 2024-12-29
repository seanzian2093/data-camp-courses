""" Use LSTM to classify sequential text data, e.g., text documents. """

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

# Define the training data - from fetch_20newsgroups of skilearn
X_train_seq = torch.tensor(
    [
        [[0.0, 0.0, 0.0, 1.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
    ]
)
y_train_seq = torch.tensor([2, 2, 2, 0, 2, 1])


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Initialize the model
model = LSTMModel(input_size=6, hidden_size=32, num_layers=2, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    optimizer.zero_grad()
    # one epoch on the entire training data
    outputs = model(X_train_seq)
    loss = criterion(outputs, y_train_seq)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch + 1}], Loss: {loss.item():.4f}")
