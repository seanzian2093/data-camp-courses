# Import the module and function

import skimage
from skimage.util import random_noise
from utils import show_image

# Prepare data
fruit_image = skimage.io.imread(
    "/users/s0046425/git_projects/2025/data-camp-courses/image_process_python/images/soaps.jpg"
)

# Add noise to the image
noisy_image = random_noise(fruit_image)

# Show original and resulting image
show_image(fruit_image, "Original")
show_image(noisy_image, "Noisy image")
