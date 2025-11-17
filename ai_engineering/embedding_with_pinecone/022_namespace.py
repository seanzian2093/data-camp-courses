import itertools

import os
from pinecone import Pinecone

with open(os.path.expanduser("~/downloads/pc.txt"), "r") as file:
    pc_token = file.read().strip()

# Set up the client with your API key
pc = Pinecone(api_key=pc_token)

index = pc.Index('datacamp-index')

# Upsert vector_set1 to namespace1
vector_set1 = [
    {
        "id": "0",
        # value should match the dimension of the index - here is for demo purpose
        "values": [0.025525547564029694, 0.0188823901116848],
        "metadata": {"genre": "action", "year": 2024},
    },
]
index.upsert(
    vectors=vector_set1,
    namespace="namespace1"
)

# Upsert vector_set2 to namespace2
vector_set2 = [
    {
        "id": "0",
        # value should match the dimension of the index - here is for demo purpose
        "values": [0.025525547564029694, 0.0188823901116848],
        "metadata": {"genre": "action", "year": 2024},
    },
]
index.upsert(
    vectors=vector_set2,
    namespace="namespace2"
)

# Print the index statistics
index.describe_index_stats()

# Query namespace1 with the vector provided
vector= [
    {
        "id": "0",
        # value should match the dimension of the index - here is for demo purpose
        "values": [0.025525547564029694, 0.0188823901116848],
        "metadata": {"genre": "action", "year": 2024},
    },
]
query_result = index.query(
    vector=vector,
    namespace="namespace1",
    top_k=3
)
print(query_result)