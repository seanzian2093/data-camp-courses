"""
Build a dataset that serves triplets consisting of 
    the image of a character to be classified, in form of file path.
    the one-hot encoded alphabet vector of length 30, ie, 0 everywhere except at the index of the alphabet.
    the target label, an integer between 0 and 963
"""

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


class OmniGlotDataset(Dataset):
    def __init__(self, transform, samples):
        self.transform = transform
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Unpack a sampel at idx
        img_path, alphabet, label = self.samples[idx]
        img = Image.open(img_path).convert("L")

        # Apply the transform
        img_transformed = self.transform(img)

        # Return a triplet
        return img_transformed, alphabet, label


# Assume we have a list of samples - a list of ('omniglot_train/Gujarati/character26/0443_16.png', 0, 10)
print(samples[100])

# Create a dataset_train
dataset_train = OmniGlotDataset(
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
        ]
    ),
    samples=samples,
)

# Create a dataloader
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
