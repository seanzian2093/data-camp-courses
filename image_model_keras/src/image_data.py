import matplotlib.pyplot as plt

# Read image data - (height, width, color channel)
# channel: 3 for rga, 4 for rgba
data = plt.imread("images/bricks.png")

# for a color image, the third dimension is the color channel(r, g, b)
# Set the red channel in this part of the image to 1
data[:10, :10, 0] = 1

# Set the green channel in this part of the image to 0
data[:10, :10, 1] = 0

# Set the blue channel in this part of the image to 0
data[:10, :10, 2] = 0

# Visualize the result
plt.imshow(data)
plt.show()
