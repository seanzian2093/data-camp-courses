""" Transforma a iamge to binary image and then separate foreground from background using thresholding """

from skimage.filters import threshold_otsu, threshold_local, try_all_threshold
from skimage import data, color
import matplotlib.pyplot as plt

from utils import show_image


# Load the rocket image
rocket = data.rocket()

# Make the image grayscale
rocket_gray = color.rgb2gray(rocket)

# Obtain the optimal threshold value with otsu - global thresholding
thresh = threshold_otsu(rocket_gray)

# Apply thresholding to the image
binary = rocket_gray > thresh

# Show the image
show_image(binary, "Binary image - global thresholding")

# Obtain the optimal threshold value with otsu - local thresholding
thresh = threshold_local(rocket_gray)

# Apply thresholding to the image
binary = rocket_gray > thresh

# Show the image
show_image(binary, "Binary image - local thresholding")

# Try all thresholding methods
fig, ax = try_all_threshold(rocket_gray, verbose=False)
plt.show()
