import itertools

import os
from pinecone import Pinecone

with open(os.path.expanduser("~/downloads/pc.txt"), "r") as file:
    pc_token = file.read().strip()


# Initialize the client allowing 20 requests in parallel
pc = Pinecone(api_key=pc_token, pool_threads=20)
index = pc.Index('datacamp-index')

def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
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

# Upsert vectors in batches of 200 vectors
with pc.Index('datacamp-index', pool_threads=20) as index:
    async_results = [index.upsert(vectors=chunk, async_req=True) for chunk in chunks(vectors, batch_size=200)]
    [async_result.get() for async_result in async_results]

# Retrieve statistics of the connected Pinecone index
print(index.describe_index_stats())