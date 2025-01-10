"""Detect faces in an image using Cascade ."""

import skimage

# from utils import show_detected_face
from skimage import data
from skimage.feature import Cascade

night_image = skimage.io.imread(
    "/users/s0046425/git_projects/2025/data-camp-courses/image_process_python/images/face_det3.jpg"
)
# Load the trained file from data
trained_file = data.lbp_frontal_face_cascade_filename()

# Initialize the detector cascade
detector = Cascade(trained_file)

# Detect faces with min and max size of searching window
detected = detector.detect_multi_scale(
    img=night_image,
    scale_factor=1.2,
    step_ratio=1,
    min_size=(10, 10),
    max_size=(200, 200),
)
print(detected)

# Show the detected faces
# show_detected_face(night_image, detected)
