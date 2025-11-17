import json
import scipy.spatial.distance as distance

bedrock_runtime = __import__("00_utils").bedrock_runtime
create_embedding = __import__("02_embed_description").create_embedding

with open("data/products_with_embeddings.json") as f:
    products = json.load(f)

# Define a function to combine the relevant features into a single string
def create_product_text(product):
  return f"""Title: {product['title']}
Description: {product['short_description']}
Category: {product['category']}
Features: {'; '.join(product['features'])}"""

# Combine the features for each product
product_texts = [create_product_text(product) for product in products]

if __name__ == "__main__":
    # Create the embeddings from product_texts
    product_embeddings = [create_embedding(text, bedrock_runtime) for text in product_texts]
    print(f"Created {len(product_embeddings)} embeddings.")
    with open("data/product_text_embeddings.json", "w") as f:
        json.dump(product_embeddings, f, indent=2)