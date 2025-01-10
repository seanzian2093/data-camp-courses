import skimage

# from utils import show_detected_face
from utils import show_image

damaged_image = skimage.io.imread(
    "/users/s0046425/git_projects/2025/data-camp-courses/image_process_python/images/sally_damaged_image.jpg"
)


# Import the necessary modules
from skimage.restoration import denoise_tv_chambolle, inpaint
from skimage.transform import rotate
import numpy as np

# Transform the image so it's not rotated
upright_img = rotate(damaged_image, 20)

# Remove noise from the image, using the chambolle method
upright_img_without_noise = denoise_tv_chambolle(
    upright_img, weight=0.1, channel_axis=2
)

# Reconstruct the image missing parts
mask_for_solution = np.zeros(damaged_image.shape[:-1])
mask_for_solution[450:475, 470:495] = 1
mask_for_solution[320:355, 140:175] = 1
mask_for_solution[130:155, 345:370] = 1
result = inpaint.inpaint_biharmonic(
    upright_img_without_noise, mask_for_solution, channel_axis=2
)

show_image(result)
