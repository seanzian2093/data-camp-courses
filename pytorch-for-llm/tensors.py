import torch

temperatures = [
    [72, 75, 78],
    [70, 73, 76],
]

# Create a tensor
temp_tensor = torch.tensor(temperatures)

if __name__ == "__main__":
    print(temp_tensor)
