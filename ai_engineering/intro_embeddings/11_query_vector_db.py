import boto3
import chromadb
from chromadb.utils.embedding_functions import AmazonBedrockEmbeddingFunction

client = chromadb.PersistentClient()
session = boto3.Session(
    region_name="us-east-1", profile_name="488608459208_BedrockPilotUsers"
)

collection = client.get_collection(
    name="netflix_titles",
    embedding_function=AmazonBedrockEmbeddingFunction(
        model_name="amazon.titan-embed-text-v1",
        session=session,
    ),
)

result = collection.query(
  query_texts=["films about dogs"],
  n_results=3
)

print(result)