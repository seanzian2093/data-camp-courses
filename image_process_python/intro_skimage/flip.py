# Import the modules from skimage
import numpy as np
from skimage import data
from utils import show_image

# Load the rocket image
rocket = data.rocket()

# Show the original image
show_image(rocket, "Original RGB image")

# Flip vertically
rocket_vertical_flip = np.flipud(rocket)
show_image(rocket_vertical_flip, "Vertically flipped image")

# Flip horizontally
rocket_horizontal_flip = np.fliplr(rocket)
show_image(rocket_horizontal_flip, "Horizontally flipped image")
