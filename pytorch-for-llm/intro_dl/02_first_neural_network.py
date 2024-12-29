import torch
import torch.nn as nn

# Must use `Tensor`, not `tensor` - but why?
input_tensor = torch.Tensor([[2, 3, 6, 7, 9, 3, 2, 1]])
# input_tensor = torch.tensor([[2, 3, 6, 7, 9, 3, 2, 1]])

# Implement a small neural network with two linear layers
model = nn.Sequential(
    # input 8-dims, output 1-dim
    nn.Linear(8, 1),
    # input 1-dims, output 1-dim
    nn.Linear(1, 1),
)

# Output
output = model(input_tensor)
print(output)
