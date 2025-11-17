import os
from pinecone import ServerlessSpec, Pinecone

with open(os.path.expanduser("~/downloads/pc.txt"), "r") as file:
    pc_token = file.read().strip()

# Initialize the Pinecone client with your API key
pc = Pinecone(api_key=pc_token)

# Create your Pinecone index
pc.create_index(
    name="my-first-index", 
    dimension=256, 
    spec=ServerlessSpec(
        cloud='aws', 
        region='us-east-1'
    )
)

print(pc)