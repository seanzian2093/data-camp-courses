""" ReLU is a non-linear activation function that is widely used in deep learning models."""

import torch
import torch.nn as nn

# Create a ReLU object
relu_pytorch = nn.ReLU()

# Apply the ReLU function to a tensor -
# requires_grad=True mean that we want to compute the gradient, i.e., the parameter is trainable, i.e., not frozen
x = torch.tensor(-1.0, requires_grad=True)
y = relu_pytorch(x)

# Compute the gradient
y.backward()

# Print the result
print(x.grad)
