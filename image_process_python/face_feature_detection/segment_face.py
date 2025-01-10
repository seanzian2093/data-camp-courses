import skimage

# from utils import show_detected_face
from skimage import data

profile_image = skimage.io.imread(
    "/users/s0046425/git_projects/2025/data-camp-courses/image_process_python/images/face_det9.jpg"
)

from skimage.segmentation import slic

# Import the label2rgb function from color module
from skimage.color import label2rgb
from skimage.feature import Cascade

# Obtain the segmentation with default 100 regions
segments = slic(profile_image)

# Obtain segmented image using label2rgb
segmented_image = label2rgb(segments, profile_image, kind="avg")

# Load the trained file from data
trained_file = data.lbp_frontal_face_cascade_filename()

# Initialize the detector cascade
detector = Cascade(trained_file)

# Detect the faces with multi scale method
detected = detector.detect_multi_scale(
    img=segmented_image,
    scale_factor=1.2,
    step_ratio=1,
    min_size=(10, 10),
    max_size=(1000, 1000),
)

# Show the detected faces
# show_detected_face(segmented_image, detected)

print(detected)
