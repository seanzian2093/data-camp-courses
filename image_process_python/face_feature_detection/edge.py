""" Use Canny algorithm to detect edges in an image """

import skimage
from utils import show_image

grapefruit = skimage.io.imread(
    "/users/s0046425/git_projects/2025/data-camp-courses/image_process_python/images/toronjas.jpg"
)
# Import the canny edge detector
from skimage.feature import canny
from skimage import color

# Convert image to grayscale
grapefruit = color.rgb2gray(grapefruit)

# Apply canny edge detector
canny_edges = canny(grapefruit)

# Show resulting image
show_image(canny_edges, "Edges with Canny")

# Apply canny edge detector with a sigma of 1.8
edges_1_8 = canny(grapefruit, sigma=1.8)

# Apply canny edge detector with a sigma of 2.2
edges_2_2 = canny(grapefruit, sigma=2.2)

# Show resulting images
show_image(edges_1_8, "Sigma of 1.8")
show_image(edges_2_2, "Sigma of 2.2")
