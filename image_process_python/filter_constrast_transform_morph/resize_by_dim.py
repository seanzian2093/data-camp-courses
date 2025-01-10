""" Resize an image by a proportional factor to a dimension """

import skimage
from utils import show_image

dogs_banner = skimage.io.imread(
    "/users/s0046425/git_projects/2025/data-camp-courses/image_process_python/images/kitty2.jpg"
)

# Import the module and function
from skimage.transform import resize

# Set proportional height so its half its size
height = int(dogs_banner.shape[0] / 2)
width = int(dogs_banner.shape[1] / 2)

# Resize using the calculated proportional height and width
image_resized = resize(dogs_banner, (height, width), anti_aliasing=True)

# Show the original and resized image
show_image(dogs_banner, "Original")
show_image(image_resized, "Resized image")
