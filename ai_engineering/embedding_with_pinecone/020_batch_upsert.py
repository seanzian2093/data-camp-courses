import itertools

import os
from pinecone import Pinecone

with open(os.path.expanduser("~/downloads/pc.txt"), "r") as file:
    pc_token = file.read().strip()

# Set up the client with your API key
pc = Pinecone(api_key=pc_token)

index = pc.Index('datacamp-index')

# To be able to batch upserts in a reproducible way, you'll need to define a function to split your list of vectors into chunks.
def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    # Convert the iterable into an iterator
    it = iter(iterable)
    # Slice the iterator into chunks of size batch_size
    # `islice` returns an iterator whose `next`returns selected elements determined by start, stop, and step parameters
    # start default to 0 and step default to 1
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        # Yield the chunk
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

# vectors should have more items - here for demo purpose
vectors = [
    {
        "id": "0",
        # value should match the dimension of the index - here is for demo purpose
        "values": [0.025525547564029694, 0.0188823901116848],
        "metadata": {"genre": "action", "year": 2024},
    },
]
# Upsert vectors in batches of 100
for chunk in chunks(vectors):
    index.upsert(vectors=chunk) 

# Retrieve statistics of the connected Pinecone index
print(index.describe_index_stats())