import matplotlib.pyplot as plt
import numpy as np

# Load the weights from file
model.load_weights("weights.hdf5")

# Predict from the first three images in the test data
model.predict(test_data[0:3])

# Get the first convolutional layer from the model
c1 = model.layers[0]

# Get the weights of the first convolutional layer
weights1 = c1.get_weights()

# Pull out the first channel of the first kernel in the first layer
kernel = weights1[0][..., 0, 0]
print(kernel)


def convolution(image, kernel):
    kernel = kernel - kernel.mean()
    result = np.zeros(image.shape)

    for ii in range(image.shape[0] - 2):
        for jj in range(image.shape[1] - 2):
            result[ii, jj] = np.sum(image[ii : ii + 2, jj : jj + 2] * kernel)

    return result


# Convolve with the fourth image in test_data
out = convolution(test_data[3, :, :, 0], kernel)

# Visualize the result
plt.imshow(out)
plt.show()
