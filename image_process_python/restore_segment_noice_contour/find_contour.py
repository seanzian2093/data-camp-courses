""" Find contours in an image """

import skimage
from utils import show_image_contour

# Prepare data
horse_image = skimage.io.imread(
    "/users/s0046425/git_projects/2025/data-camp-courses/image_process_python/images/horse.png"
)

# Import the modules
from skimage import data, measure

# Obtain the horse image
horse_image = data.horse()

# Find the contours with a constant level value of 0.8
contours = measure.find_contours(horse_image, 0.8)


# Shows the image with contours found
show_image_contour(horse_image, contours)
