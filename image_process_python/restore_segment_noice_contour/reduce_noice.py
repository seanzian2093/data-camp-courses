# Import the module and function

import skimage
from skimage.restoration import denoise_tv_chambolle
from utils import show_image

# Prepare data
noisy_image = skimage.io.imread(
    "/users/s0046425/git_projects/2025/data-camp-courses/image_process_python/images/soaps.jpg"
)

# Apply total variation filter denoising
denoised_image = denoise_tv_chambolle(noisy_image, channel_axis=2)

# Show the noisy and denoised images
show_image(noisy_image, "Noisy")
show_image(denoised_image, "Denoised image")
