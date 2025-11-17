import os
from pinecone import Pinecone

with open(os.path.expanduser("~/downloads/pc.txt"), "r") as file:
    pc_token = file.read().strip()

# Set up the client with your API key
pc = Pinecone(api_key=pc_token)

index = pc.Index('datacamp-index')
ids = ['2', '5', '8']

# Fetch vectors from the connected Pinecone index
fetched_vectors = index.fetch(ids=ids)

# Extract the metadata from each result in fetched_vectors
metadatas = [fetched_vectors['vectors'][id]['metadata'] for id in ids]
print(metadatas)

# Retrieve the top three most records that are similar to a given vector
vector = {
        "id": "0",
        # value should match the dimension of the index - here is for demo purpose
        "values": [0.025525547564029694, 0.0188823901116848],
        "metadata": {"genre": "action", "year": 2024},
    }

query_result = index.query(
    vector=vector,
    top_k=3
)

print(query_result)

# Print a list of your indexes
print(pc.list_indexes())

# Retrieve the MOST similar vector with the year 2024
query_result = index.query(
    vector=vector,
    top_k=1,
    filter={
        "year": 2024
    }
)
print(query_result)

# Retrieve the MOST similar vector with genre and year filters
query_result = index.query(
    vector=vector,
    top_k=1,
    filter={
        "genre": "thriller",
        "year": {"$lt": 2018}
    }
)
print(query_result)

# Update the values of vector ID 7
index.update(
    id="7",
    values=vector
)

# Fetch vector ID 7
fetched_vector = index.fetch(ids=['7'])
print(fetched_vector)


# Update the metadata of vector ID 7
index.update(
    id="7",
    set_metadata={
        "genre": "thriller", 
        "year": 2024} 
)

# Fetch vector ID 7
fetched_vector = index.fetch(ids=['7'])
print(fetched_vector)

# Delete vectors
index.delete(
  ids=["3", "4"]
)

# Retrieve metrics of the connected Pinecone index
print(index.describe_index_stats())