import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

# Read image data
im = plt.imread("images/bricks.png")
print(im.shape)
# Convert the image to grayscale - keep only `rgb` channels
im = im[:, :, 0:3]
im = rgb2gray(im)
print(im.shape)

# 1D Convolution
array = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
kernel = np.array([1, -1, 0])
conv = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Output array - 8 = len(array) - len(kernel) + 1
for ii in range(8):
    # move kernel along array, one elm at a time, and element-wise multiply and sum
    conv[ii] = (kernel * array[ii : ii + 3]).sum()

# Print conv
print(conv)


# 2D Convolution - image convolutions
kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
result = np.zeros(im.shape)

# Output array
for ii in range(im.shape[0] - 3):
    for jj in range(im.shape[1] - 3):
        result[ii, jj] = (im[ii : ii + 3, jj : jj + 3] * kernel).sum()

# Print result
print(result)

# Typical kernels - -1 is dark pixel, 1 is bright pixel
# a kernel that finds horizontal lines in iamges - dark pixels at top, bright in middle, dark at bottom
kernel = np.array([[-1, -1, -1], [1, 1, 1], [-1, -1, -1]])

# a kernel that finds a light spot surrounded by dark pixels.
kernel = np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]])
# a kernel that finds a dark spot surrounded by bright pixels.
kernel = np.array([[1, 1, 1], [1, -1, 1], [1, 1, 1]])
