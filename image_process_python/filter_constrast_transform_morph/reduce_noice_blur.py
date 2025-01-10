""" Use the GaussianBlur filter to reduce noise in the image. """

import skimage
from utils import show_image

building_image = skimage.io.imread(
    "/users/s0046425/git_projects/2025/data-camp-courses/image_process_python/images/soaps.jpg"
)
# Import Gaussian filter
from skimage.filters import gaussian

# Apply filter - multichannel is depreciated since version 0.18, this is 0.25
# gaussian_image = gaussian(building_image, multichannel=True)
print(building_image.shape)
gaussian_image = gaussian(building_image, channel_axis=2)

# Show original and resulting image to compare
show_image(building_image, "Original")
show_image(gaussian_image, "Reduced sharpness Gaussian")
