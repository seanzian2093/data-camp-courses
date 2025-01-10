""" Improve the contrast of an low contrast, ie., x-ray image """

import skimage
from utils import show_image
import matplotlib.pyplot as plt

chest_xray_image = skimage.io.imread(
    "/users/s0046425/git_projects/2025/data-camp-courses/image_process_python/images/contrast_00000109_005.png"
)
# Import the required module
from skimage import exposure

# Show original x-ray image and its histogram
show_image(chest_xray_image, "Original x-ray")

plt.title("Histogram of image")
plt.hist(chest_xray_image.ravel(), bins=256)
plt.show()

# Use histogram equalization to improve the contrast
xray_image_eq = exposure.equalize_hist(chest_xray_image)

# Show the resulting image
plt.title("Histogram of equalized image")
plt.hist(xray_image_eq.ravel(), bins=256)
plt.show()

show_image(xray_image_eq, "Resulting image")
