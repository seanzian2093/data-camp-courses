""" Use bilateral filter to reduce noise while preserving edges """

# Import bilateral denoising function
import skimage
from utils import show_image

# Prepare data
landscape_image = skimage.io.imread(
    "/users/s0046425/git_projects/2025/data-camp-courses/image_process_python/images/noise-noisy-nature.jpg"
)
from skimage.restoration import denoise_bilateral

# Apply bilateral filter denoising
denoised_image = denoise_bilateral(landscape_image, channel_axis=2)

# Show original and resulting images
show_image(landscape_image, "Noisy image")
show_image(denoised_image, "Denoised image")
