""" Use dilation morphological operation to improve the thresholded image. """

import skimage
from utils import show_image

world_image = skimage.io.imread(
    "/users/s0046425/git_projects/2025/data-camp-courses/image_process_python/images/r5.png"
)

# Import the module
from skimage import morphology

# Obtain the dilated image
dilated_image = morphology.binary_dilation(world_image)

# See results
show_image(world_image, "Original")
show_image(dilated_image, "Dilated image")
