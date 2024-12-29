import torch
import torch.nn as nn

# Map a unique index/integer to each word
words = [
    "This",
    "book",
    "was",
    "fantastic",
    "I",
    "really",
    "love",
    "science",
    "fiction",
    "but",
    "the",
    "protagonist",
    "was",
    "rude",
    "sometimes",
]

word_to_idx = {word: i for i, word in enumerate(words)}

# Convert word_to_idx to a tensor - a LongTensor which is required by nn.Embedding.
# `torch.Tensor` creates a float tensor by default
inputs = torch.LongTensor([word_to_idx[w] for w in words])


# Initialize the embedding layer with 10 dimensions - 10 is arbitrary here
embedding = nn.Embedding(num_embeddings=len(words), embedding_dim=10)

# Pass the tensor to the embedding layer
output = embedding(inputs)
print(output)
