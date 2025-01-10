import skimage

# from utils import show_detected_face
from skimage import data
from skimage.feature import Cascade
from skimage.filters import gaussian
from utils import getFaceRectangle, mergeBlurryFace, show_image

group_image = skimage.io.imread(
    "/users/s0046425/git_projects/2025/data-camp-courses/image_process_python/images/face_det25.jpg"
)

# Load the trained file from data
trained_file = data.lbp_frontal_face_cascade_filename()

# Initialize the detector cascade
detector = Cascade(trained_file)

# Detect the faces
detected = detector.detect_multi_scale(
    img=group_image,
    scale_factor=1.2,
    step_ratio=1,
    min_size=(10, 10),
    max_size=(100, 100),
)
# For each detected face
for d in detected:
    # Obtain the face rectangle from detected coordinates
    face = getFaceRectangle(d, group_image)

    # Apply gaussian filter to extracted face
    blurred_face = gaussian(face, channel_axis=2, sigma=8)

    # Merge this blurry face to our final image and show it
    resulting_image = mergeBlurryFace(d, group_image, blurred_face)
show_image(resulting_image, "Blurred faces")
