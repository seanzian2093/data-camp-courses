import numpy as np

# Pooling layers are often added between the convolutional layers of a neural network to summarize their outputs in a condensed manner,
# and reduce the number of parameters in the next layer in the network. This can help us if we want to train the network more rapidly,
# or if we don't have enough data to learn a very large number of parameters.

# A pooling layer can be described as a particular kind of convolution.
# For every window in the input it finds the maximal pixel value and passes only this pixel through.

# Result placeholder
result = np.zeros((im.shape[0] // 2, im.shape[1] // 2))

# Pooling operation - get the max value in each 2x2 window
for ii in range(result.shape[0]):
    for jj in range(result.shape[1]):
        result[ii, jj] = np.max(im[ii * 2 : ii * 2 + 2, jj * 2 : jj * 2 + 2])
