""" PyTorch Dataset and DataLoader example using class approach """

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class RandomDataset(Dataset):
    def __init__(self, num_samples, num_features):
        self.num_samples = num_samples
        self.num_features = num_features
        self.data = np.random.randn(num_samples, num_features)
        self.labels = np.random.randint(0, 2, num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(
            label, dtype=torch.long
        )


# Example usage
if __name__ == "__main__":
    dataset = RandomDataset(num_samples=1000, num_features=10)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # loop over all the data
    for data, labels in dataloader:
        print(data, labels)
        break

    # get a batch of data
    data, labels = next(iter(dataloader))
    print(data, labels)
