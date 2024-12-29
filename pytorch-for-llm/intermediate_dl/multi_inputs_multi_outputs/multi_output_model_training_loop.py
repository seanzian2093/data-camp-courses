import torch
import torch.nn as nn
from two_input_dataset import dataloader_train
from two_output_model import Net

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

for epoch in range(3):
    for images, labels_alpha, labels_char in dataloader_train:
        optimizer.zero_grad()
        outputs_alpha, outputs_char = net(images)
        # Compute the loss - alphabet classifier
        loss_alpha = criterion(outputs_alpha, labels_alpha)
        # Compute the loss - character classifier
        loss_char = criterion(outputs_char, labels_char)
        # Compute total loss
        loss = loss_alpha + loss_char
        loss.backward()
        optimizer.step()
