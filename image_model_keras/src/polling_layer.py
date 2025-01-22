# Import the necessary components from Keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D

img_rows, img_cols = 28, 28

# Initialize the model object
model = Sequential()

# Add a convolutional layer
# number of parameters = (kernel_height * kernel_width * input_channels + 1) * number_of_filters
# 1 for bias term
model.add(
    Conv2D(15, kernel_size=2, activation="relu", input_shape=(img_rows, img_cols, 1))
)

# Add a pooling operation
model.add(MaxPool2D(pool_size=2))

# Add another convolutional layer -
# should remove the input_shape parameter?
# because it should take the output shape of the previous layer as its input?
model.add(
    # Conv2D(5, kernel_size=2, activation="relu", input_shape=(img_rows, img_cols, 1))
    # this layer take previous layer output shape as input so input_channel = 15
    Conv2D(5, kernel_size=2, activation="relu")
)

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation="softmax"))
model.summary()
