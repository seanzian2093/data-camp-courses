import torch

genres = ["Fiction", "Non-Fiction", "Biography", "Children", "Mystery"]

# Define the size of the vocabulary
vocab_size = len(genres)

# Create one-hot vectors - `eye` creates a 2-D tensor with ones on the diagonal and zeros elsewhere
one_hot_vectors = torch.eye(vocab_size)

# Create a dictionary to map genres to one-hot vectors
one_hot_dict = {genre: one_hot_vectors[i] for i, genre in enumerate(genres)}
for genre, one_hot_vector in one_hot_dict.items():
    print(f"{genre}: {one_hot_vector.numpy()}")
