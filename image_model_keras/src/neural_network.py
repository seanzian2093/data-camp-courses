# Imports components from Keras
from keras.models import Sequential
from keras.layers import Dense

# Initializes a sequential model
model = Sequential()

# First layer
# Dense means each unit in each layer is connected to all of the units in the previous layer
model.add(Dense(10, activation="relu", input_shape=(784,)))

# Second layer
model.add(Dense(10, activation="relu"))

# Output layer
model.add(Dense(units=3, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Reshape the data to two-dimensional array
# `train_data` is originally of shape (50, 28, 28, 1)
# 50 images, 28x28 pixels, 1 color channel
train_data = train_data.reshape(50, 28 * 28)

# Fit the model
# `train_labels` is of shape(50, 3), i.e. 50 images, one-hot-encoded classes
model.fit(train_data, train_labels, validation_split=0.2, epochs=3)

# Reshape test data - 10 images of 28x28 pixels
test_data = test_data.reshape(10, 28 * 28)

# Evaluate the model
model.evaluate(test_data, test_labels)
