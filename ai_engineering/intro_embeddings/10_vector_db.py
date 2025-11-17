import boto3
import csv
import chromadb
from chromadb.utils.embedding_functions import AmazonBedrockEmbeddingFunction

ids = []
documents = []

with open("data/netflix_titles.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader):
        ids.append(row["show_id"])
        text = f"Title: {row['title']} ({row['type']})\nDescription: {row['description']}\nCategories: {row['listed_in']}"
        documents.append(text)

# Create a persistant client
client = chromadb.PersistentClient()
session = boto3.Session(
    region_name="us-east-1", profile_name="488608459208_BedrockPilotUsers"
)

# Delete existing collection if it exists
try:
    client.delete_collection(name="netflix_titles")
    print("Existing collection deleted")
except ValueError:
    print("No existing collection to delete")

# Create a netflix_title collection using the Bedrock Embedding function
collection = client.create_collection(
    name="netflix_titles",
    embedding_function=AmazonBedrockEmbeddingFunction(
        model_name="amazon.titan-embed-text-v1",
        session=session,
    ),
)

# List the collections
print(client.list_collections())

# This takes a long time to run ~30 minutes
if __name__ == "__main__":
    # Adjust this size as needed - this is a limit to which `collection.add()` can handle at one time
    batch_size = 1000  

    for i in range(0, len(documents), batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_documents = documents[i : i + batch_size]

        print(
            f"Adding batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}"
        )
        # Add the documents and IDs to the collection in batches
        collection.add(ids=batch_ids, documents=batch_documents)

    # Print the collection size and first ten items
    print(f"No. of documents: {collection.count()}")
    print(f"First ten documents: {collection.peek()}")
