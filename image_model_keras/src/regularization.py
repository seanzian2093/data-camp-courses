# Import the necessary components from Keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization

img_rows, img_cols = 28, 28
# Initialize the model object
model = Sequential()

# Add a convolutional layer
model.add(
    Conv2D(15, kernel_size=2, activation="relu", input_shape=(img_rows, img_cols, 1))
)

# Add a dropout layer
# Dropout is a form of regularization that removes a different random subset of the units in a layer in each round of training.
model.add(Dropout(0.2))

# Batch normalization is another form of regularization that rescales the outputs of a layer to make sure that they have mean 0 and standard deviation 1.
# Add batch normalization layer
model.add(BatchNormalization())

# Add another convolutional layer
model.add(Conv2D(5, kernel_size=2, activation="relu"))

# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(3, activation="softmax"))
