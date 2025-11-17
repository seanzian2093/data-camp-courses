import os
from pinecone import Pinecone

with open(os.path.expanduser("~/downloads/pc.txt"), "r") as file:
    pc_token = file.read().strip()

# Set up the client with your API key
pc = Pinecone(api_key=pc_token)

# Connect to your index
index = pc.Index("my-first-index")

# Print the index statistics
print(index.describe_index_stats())

# Delete your Pinecone index
pc.delete_index("my-first-index")

# List your indexes
print(pc.list_indexes())

# Connect to your index
index = pc.Index("datacamp-index")

vectors = [
    {
        "id": "0",
        # value should match the dimension of the index - here is for demo purpose
        "values": [0.025525547564029694, 0.0188823901116848],
        "metadata": {"genre": "action", "year": 2024},
    },
]
# Ingest the vectors and metadata
index.upsert(
    vectors=vectors
)

# Print the index statistics
print(index.describe_index_stats())