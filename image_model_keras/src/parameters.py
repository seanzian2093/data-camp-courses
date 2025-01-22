# Import the necessary components from Keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

# CNN model
model = Sequential()
model.add(Conv2D(10, kernel_size=2, activation="relu", input_shape=(28, 28, 1)))
model.add(Conv2D(10, kernel_size=2, activation="relu"))
model.add(Flatten())
model.add(Dense(3, activation="softmax"))

# Summarize the model
model.summary()
