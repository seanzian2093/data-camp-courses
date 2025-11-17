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

reference_texts = ["children's story about a car", "lions"]

# Query two results using reference_texts
result = collection.query(
  query_texts=reference_texts,
  n_results=2,
  # Filter for titles with a G rating and released before 2019
  where={
    "$and": [
        {"rating": 
        	{"$eq": "G"}
        },
        {"release_year": 
         	{"$lt": 2019}
        }
    ]
  }
)

print(result['documents'])
