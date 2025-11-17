import json
import scipy.spatial.distance as distance

bedrock_runtime = __import__("00_utils").bedrock_runtime
create_embedding = __import__("02_embed_description").create_embedding
create_product_text = __import__("05_embedding_for_ai").create_product_text

with open("data/products_with_embeddings.json") as f:
    products = json.load(f)

with open("data/product_text_embeddings.json", "r") as f:
    product_embeddings = json.load(f)

def find_n_closest(query_vector, embeddings, n=3):
    distances = []
    for index, embedding in enumerate(embeddings):
    # Calculate the cosine distance between the query vector and embedding
        dist = distance.cosine(query_vector, embedding)
    # Append the distance and index to distances
        distances.append({"distance": dist, "index": index})
    # Sort distances by the distance key
    distances_sorted = sorted(distances, key=lambda x: x["distance"])
    # Return the first n elements in distances_sorted
    return distances_sorted[0:n]

# Create the query vector from query_text
query_text = "computer"
query_vector = create_embedding(query_text, bedrock_runtime)

# Find the five closest distances
hits = find_n_closest(query_vector, product_embeddings, n=5)

print(f'Search results for "{query_text}"')
for hit in hits:
    # Extract the product at each index in hits
    product = products[hit['index']]
    print(product["title"])

# Combine the features for last_product and each product in products
last_product = {'title': 'Building Blocks Deluxe Set',
 'short_description': 'Unleash your creativity with this deluxe set of building blocks for endless fun.',
 'price': 34.99,
 'category': 'Toys',
 'features': ['Includes 500+ colorful building blocks',
  'Promotes STEM learning and creativity',
  'Compatible with other major brick brands',
  'Comes with a durable storage container',
  'Ideal for children ages 3 and up']}

last_product_text = create_product_text(last_product)
last_product_embeddings = create_embedding(last_product_text, bedrock_runtime)


# Find the three smallest cosine distances and their indexes
hits = find_n_closest(last_product_embeddings, product_embeddings)

print(f'\nSearch results for "{last_product["title"]}"')
for hit in hits:
  product = products[hit['index']]
  print(product['title'])