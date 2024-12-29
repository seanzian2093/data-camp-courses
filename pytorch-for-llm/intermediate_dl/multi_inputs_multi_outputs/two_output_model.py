""" One input, two output"""

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Define sub-networks as sequential models
        self.image_layer = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(16 * 32 * 32, 128),
        )

        # Define the two classifier layers
        # output is 30 because the alphabet has 30 characters
        self.classifier_alpha = nn.Sequential(nn.Linear(128, 30))
        # output is 964 because there are 964 characters in the dataset
        self.classifier_char = nn.Sequential(nn.Linear(128, 964))

    def forward(self, x):
        # Pass x through the image layer
        x_image = self.image_layer(x)
        # Pass the output of the image layer through the two classifiers
        output_alpha = self.classifier_alpha(x_image)
        output_char = self.classifier_char(x_image)
        return output_alpha, output_char
