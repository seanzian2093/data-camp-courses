import json
import numpy as np
import scipy.spatial.distance as distance

bedrock_runtime = __import__("00_utils").bedrock_runtime
create_embedding = __import__("02_embed_description").create_embedding

with open("data/products_with_embeddings.json") as f:
    products = json.load(f)

# Embed the search text
search_text = "soap"
search_embedding = create_embedding(search_text, bedrock_runtime)

distances = []
for product in products:
  # Compute the cosine distance for each product description
  dist = distance.cosine(search_embedding, product["embedding"])
  distances.append(dist)

# Find and print the most similar product short_description    
min_dist_ind = np.argmin(distances)
print(products[min_dist_ind]['short_description'])