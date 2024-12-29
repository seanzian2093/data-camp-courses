import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.feature_extractor = nn.Sequential(
            # output size should be input size of next layer of Conv2d
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            # 64 is the dimension of the output of the feature_extractor
            # height and width divided by `kernel_size` of every layer of Conv2d - 64/2/2 and 64/2/2
            # since we `Flatten()` so input size should be 64 * 16 * 16
            # output size should be equal to `num_classes`
            nn.Linear(in_features=64 * 16 * 16, out_features=num_classes),
        )

    def forward(self, x):
        # Pass input through the feature extractor and then the classifier
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
