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

new_data = [{"id": "s1001", "document": "Title: Cats & Dogs (Movie)\nDescription: A look at the top-secret, high-tech espionage war going on between cats and dogs, of which their human owners are blissfully unaware."},
 {"id": "s6884", "document": 'Title: Goosebumps 2: Haunted Halloween (Movie)\nDescription: Three teens spend their Halloween trying to stop a magical book, which brings characters from the "Goosebumps" novels to life.\nCategories: Children & Family Movies, Comedies'}]

# Update or add the new documents
collection.upsert(
    ids=[doc['id'] for doc in new_data],
    documents=[doc['document'] for doc in new_data]
)

# Delete the item with ID "s95" and re-run the query
collection.delete(ids=["s95"])

result = collection.query(
    query_texts=["films about dogs"],
    n_results=3
)

print(result['documents'])


# Qeury by text
reference_ids = ['s999', 's1000']

# Retrieve the documents for the reference_ids
reference_texts = collection.get(ids=reference_ids)['documents']

# Query using reference_texts
result = collection.query(
  query_texts=reference_texts,
  n_results=3
)

print("\n", result['documents'])

reference_texts = ["children's story about a car", "lions"]

# Query two results using reference_texts
result = collection.query(
  query_texts=reference_texts,
  n_results=2,
  # Filter for titles with a G rating released before 2019
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