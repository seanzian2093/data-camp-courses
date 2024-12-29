"""
The Leaky ReLU activation function is a variant of the Rectified Linear Unit (ReLU) that allows a small, non-zero gradient when the input is negative.
"""

import torch
import torch.nn as nn

# Define the Leaky ReLU activation function
leaky_relu = nn.LeakyReLU(negative_slope=0.05)

# Example input tensor
input_tensor = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

# Apply the Leaky ReLU activation function
output_tensor = leaky_relu(input_tensor)

print("Input Tensor:", input_tensor)
print("Output Tensor:", output_tensor)
