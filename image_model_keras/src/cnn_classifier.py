# Import the necessary components from Keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

# Initialize the model object
model = Sequential()

# Add a convolutional layer
# 1st argument, units, specifies the number of filters (or kernels) that the convolutional layer will learn. Each filter will produce a feature map, and the number of filters determines the depth of the output volume.
# Padding allows a convolutional layer to retain the resolution of the input into this layer. This is done by adding zeros around the edges of the input image, so that the convolution kernel can overlap with the pixels on the edge of the image.
# The size of the strides of the convolution kernel determines whether the kernel will skip over some of the pixels as it slides along the image. This affects the size of the output because when strides are larger than one, the kernel will be centered on only some of the pixels.
model.add(
    Conv2D(
        10,
        kernel_size=3,
        activation="relu",
        input_shape=(img_rows, img_cols, 1),
        padding="same",
        strides=2,
    )
)

# Flatten the output of the convolutional layer - multi-dimensional to one-dimensional
# typicall for the output of a convolutional layer to be flattened before passing it to a dense layer
model.add(Flatten())
# Add an output layer for the 3 categories
# 1st argument, units, specifies the number of units (or neurons) in that layer. This determines the dimensionality of the output space.
model.add(Dense(3, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Fit the model on a training set
# `batch_size` specifies the number of samples that will be propagated through the network at once.
model.fit(train_data, train_labels, validation_split=0.2, epochs=3, batch_size=10)

# Evaluate the model on separate test data
# we can use different batch size for evaluation from the one used for training
model.evaluate(test_data, test_labels, batch_size=10)
