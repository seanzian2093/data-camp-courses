""" Detect edges in an image by applying Sobel filter."""

import skimage
from utils import show_image

soaps_image = skimage.io.imread(
    "/users/s0046425/git_projects/2025/data-camp-courses/image_process_python/images/soaps.jpg"
)

# Import the color module
from skimage import color

# Import the filters module and sobel function
from skimage.filters import sobel

# Make the image grayscale
soaps_image_gray = color.rgb2gray(soaps_image)

# Apply edge detection filter
edge_sobel = sobel(soaps_image_gray)

# Show original and resulting image to compare
show_image(soaps_image, "Original")
show_image(edge_sobel, "Edges with Sobel")
