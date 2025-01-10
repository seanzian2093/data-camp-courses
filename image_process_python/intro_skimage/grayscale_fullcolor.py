""" Use `scikit-image` 
`pip3 install scikit-image` to install the package
"""

from skimage import data

coffee_image = data.coffee()
coins_image = data.coins()

print(coffee_image.shape)
# (400, 600, 3) is (height, width, channels), where channels are R, G, B
print(coins_image.shape)
# (303, 384) is (height, width)
