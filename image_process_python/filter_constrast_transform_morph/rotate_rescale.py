""" Rotate and rescale the image """

import skimage
from utils import show_image

image_cat = skimage.io.imread(
    "/users/s0046425/git_projects/2025/data-camp-courses/image_process_python/images/kitty2.jpg"
)

# Import the module and the rotate and rescale functions
from skimage.transform import rotate, rescale

# Rotate the image 90 degrees clockwise
rotated_cat_image = rotate(image_cat, -90)

# Rescale with anti aliasing - anti aliasing is for smoothing the image
print(rotated_cat_image.shape)
rescaled_with_aa = rescale(rotated_cat_image, 1 / 4, anti_aliasing=True, channel_axis=2)

# Rescale without anti aliasing
rescaled_without_aa = rescale(
    rotated_cat_image, 1 / 4, anti_aliasing=False, channel_axis=2
)

# Show the resulting images
show_image(rescaled_with_aa, "Transformed with anti aliasing")
show_image(rescaled_without_aa, "Transformed without anti aliasing")
