""" Use morphological erosion to discard pixels near boundaries from a binary image, good for OCR. """

import skimage
from utils import show_image

upper_r_image = skimage.io.imread(
    "/users/s0046425/git_projects/2025/data-camp-courses/image_process_python/images/r5.png"
)

# Import the morphology module
from skimage import morphology

# Obtain the eroded shape
eroded_image_shape = morphology.binary_erosion(upper_r_image)

# See results
show_image(upper_r_image, "Original")
show_image(eroded_image_shape, "Eroded image")
