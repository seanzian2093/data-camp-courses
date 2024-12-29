from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define a series of transformations - augmentations
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomAutocontrast(),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((64, 64)),
    ]
)

# Create a dataset
dataset = datasets.ImageFolder(root="images", transform=transform)

# Example of how to use the dataset with a DataLoader

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through the DataLoader
for images, labels in dataloader:
    # Your training code here
    pass

# Reshape the images to be displayed
image, label = next(iter(dataloader))
image = image.squeeze(0).permute(1, 2, 0)

# Display the image
plt.imshow(image)
plt.show()
