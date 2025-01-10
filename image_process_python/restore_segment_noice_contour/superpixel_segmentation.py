""" Use SLIC superpixel segmentation to segment the image into regions - unsupervised machine learning technique """

import skimage
from utils import show_image

# Prepare data
face_image = skimage.io.imread(
    "/users/s0046425/git_projects/2025/data-camp-courses/image_process_python/images/chinese.jpg"
)

# Import the slic function from segmentation module
from skimage.segmentation import slic

# Import the label2rgb function from color module
from skimage.color import label2rgb

# Obtain the segmentation with 400 regions
segments = slic(face_image, n_segments=400)

# a labeled image (where each region is represented by a unique integer label) into an RGB image.
# Put segments on top of original image to compare
# The method to use for coloring the labels.
# The "avg" option means that each segment will be colored with the average color of the pixels in that segment from the original image
segmented_image = label2rgb(segments, face_image, kind="avg")

# Show the segmented image
show_image(segmented_image, "Segmented image, 400 superpixels")
