""" Two input, one output"""

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Define sub-networks as sequential models
        self.image_layer = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.MaxPool2d(2),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(16 * 32 * 32, 128),
        )

        self.alphabet_layer = nn.Sequential(
            nn.Linear(30, 8),
            nn.ELU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 + 8, 964),
        )

    def forward(self, x_image, x_alphabet):
        # Pass x_image through the image layer
        x_image = self.image_layer(x_image)
        # Pass x_alphabet through the alphabet layer
        x_alphabet = self.alphabet_layer(x_alphabet)
        # Concatenate the outputs of the two sub-networks
        x = torch.cat([x_image, x_alphabet], dim=1)
        # Pass the concatenated output through the classifier
        x = self.classifier(x)
        return x
