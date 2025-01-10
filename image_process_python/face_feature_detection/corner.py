""" Detect corners in an image using Harris corner detection algorithm. """

import skimage
from utils import show_image, show_image_with_corners

building_image = skimage.io.imread(
    "/users/s0046425/git_projects/2025/data-camp-courses/image_process_python/images/corners_building_top.jpg"
)
# Import the corner detector related functions and module
from skimage.feature import corner_harris, corner_peaks
from skimage import color

# Convert image from RGB-3 to grayscale
building_image_gray = color.rgb2gray(building_image)

# Apply the detector  to measure the possible corners
measure_image = corner_harris(building_image_gray)

# Find the peaks of the corners using the Harris detector
# threshold_rel specifies the minimum intensity of the corner
# and min_distance is the minimum number of pixels separating
coords = corner_peaks(measure_image, min_distance=20, threshold_rel=0.02)
print(
    "With a min_distance set to 20, we detect a total",
    len(coords),
    "corners in the image.",
)

# Show original and resulting image with corners detected
show_image(building_image, "Original")
show_image_with_corners(building_image, coords)

# Find the peaks with a min distance of 10 pixels
coords_w_min_10 = corner_peaks(measure_image, min_distance=10, threshold_rel=0.02)
print(
    "With a min_distance set to 10, we detect a total",
    len(coords_w_min_10),
    "corners in the image.",
)
