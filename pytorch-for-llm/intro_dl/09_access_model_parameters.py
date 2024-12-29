""" Access layers and parameters of a model in PyTorch"""

import torch
import torch.nn as nn

# input_tensor shape (1, 11), i.e., input dimension = 11
input_tensor = torch.Tensor([[3, 4, 6, 7, 10, 12, 2, 3, 6, 8, 9]])

# first layer must accept 11-dim
model = nn.Sequential(
    nn.Linear(11, 20),
    # previous layer's output dim = current layer's input dim
    nn.Linear(20, 12),
    nn.Linear(12, 6),
    nn.Linear(6, 4),
)

# layers of nn.Sequential are 0-indexed

# weight of a layer
weight_0 = model[0].weight
print("Weight of 1st layer:", weight_0)

# gradients of a weight
grads_0 = weight_0.grad
print("Gradients of weight of 1st layer:", grads_0)

# Update the weight manually
lr = 0.1
weight_0 = weight_0 - lr * grads_0

bias_1 = model[1].bias
print("Bias of 2nd layer:", bias_1)
