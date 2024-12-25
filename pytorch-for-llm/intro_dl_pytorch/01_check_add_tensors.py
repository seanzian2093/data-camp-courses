import torch

temperatures = torch.tensor(
    [
        [72, 75, 78],
        [70, 73, 76],
    ]
)

adjustment = torch.tensor(
    [
        [2, 2, 2],
        [2, 2, 2],
    ]
)

# Type
temp_type = temperatures.dtype
print("Data type of temperatures:", temp_type)

# Shape
temp_shape = temperatures.shape
print("Shape of temperatures:", temp_shape)

# Adding
corrected_temp = temperatures + adjustment
print("Corrected temperatures:", corrected_temp)
