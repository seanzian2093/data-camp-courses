""" Access the number of parameters in a model in PyTorch"""

import torch.nn as nn

# Define a simple sequential model
model = nn.Sequential(nn.Linear(16, 4), nn.Linear(4, 2), nn.Linear(2, 1))

# Count the number of parameters using a for loop
num_params = 0
for param in model.parameters():
    if param.requires_grad:
        num_params += param.numel()

# Print the number of parameters
print(f"The model has {num_params} trainable parameters")
