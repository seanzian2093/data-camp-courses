import torch.nn as nn


class NetWithInit(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        # Add normalization to the weights
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, output_size)

        # Custom initialization
        nn.init.kaiming_uniform_(self.layer1.weight)
        nn.init.kaiming_uniform_(self.layer2.weight)
        nn.init.kaiming_uniform_(self.layer3.weight, nonlinearity="sigmoid")

    def forward(self, x):
        x = self.layer1(x)
        # Pass through the batch normalization layer
        x = self.bn1(x)
        # Use ELU activation function
        x = nn.functional.elu(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = nn.functional.elu(x)

        x = nn.functional.sigmoid(self.layer3(x))
        return x
