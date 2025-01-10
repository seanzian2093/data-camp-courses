""" Restore the damaged image with missng parts. """

import skimage
import numpy as np
from skimage.transform import resize
from skimage.restoration import inpaint
from utils import show_image

# Prepare data
defect_image = skimage.io.imread(
    "/users/s0046425/git_projects/2025/data-camp-courses/image_process_python/images/damaged_astronaut.png"
)
# Resize the image to (512, 512)
defect_image_resized = resize(defect_image, (512, 512), anti_aliasing=True)


# mask shoud be a binary image with the same shape as the image to restore
# mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
# Create mask with six block defect regions
mask = np.zeros(defect_image_resized.shape[:-1], dtype=bool)
mask[20:60, 0:20] = 1
mask[160:180, 70:155] = 1
mask[30:60, 170:195] = 1
mask[-60:-30, 170:195] = 1
mask[-180:-160, 70:155] = 1
mask[-60:-20, 0:20] = 1

# Add a few long, narrow defects
mask[200:205, -200:] = 1
mask[150:255, 20:23] = 1
mask[365:368, 60:130] = 1

# Import the module from restoration
from skimage.restoration import inpaint

# Show the defective image
show_image(defect_image_resized, "Image to restore")
print(defect_image_resized.shape)

show_image(mask, "Mask")
print(mask.shape)
# Apply the restoration function to the image using the mask
restored_image = inpaint.inpaint_biharmonic(defect_image_resized, mask, channel_axis=-1)
print(restored_image.shape)
show_image(restored_image)
