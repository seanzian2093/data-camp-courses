"""
Demonstrates the use of Sigmoid and Softmax activation functions in PyTorch for binary and multi-class classification.
"""

import torch
import torch.nn as nn

input_tensor = torch.tensor([[0.8]])

# Output 1-dim - for binary classification
sigmoid = nn.Sigmoid()
probability = sigmoid(input_tensor)
print(probability)

# Output n-dim - for multi-class classification
input_tensor2 = torch.tensor([[1.0, -6.0, 2.5, -0.3, 1.2, 0.8]])
# input_tensor2 shape is [1, 2], i.e., 2 dimensional, so dim = 1
softmax = nn.Softmax(dim=1)
probabilities = softmax(input_tensor2)
print(probabilities)
