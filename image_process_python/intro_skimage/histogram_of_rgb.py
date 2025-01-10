import numpy as np
from skimage import data
import matplotlib.pyplot as plt

# Load the rocket image
image = data.rocket()

# Obtain the red channel - RGB is the last dimension of 3, i.e., 0 for red, 1 for green, 2 for blue
red_channel = image[:, :, 0]

# Plot the red histogram with bins in a range of 256
plt.hist(red_channel.ravel(), bins=256)

# Set title and show
plt.title("Red Histogram")
plt.show()
