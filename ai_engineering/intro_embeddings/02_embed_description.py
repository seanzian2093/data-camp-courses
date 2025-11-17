import json
import time

utils = __import__("00_utils")
bedrock_runtime = utils.bedrock_runtime

with open("data/products.json") as f:
    products = json.load(f)


def create_embedding(text, bedrock_runtime):
    body = json.dumps({"inputText": text})

    response = bedrock_runtime.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        body=body,
        contentType="application/json",
    )

    response_body = json.loads(response["body"].read())
    embedding = response_body["embedding"]

    return embedding


# Process embeddings with progress tracking and error handling
if __name__ == "__main__":
    for i, product in enumerate(products):
        try:
            product["embedding"] = create_embedding(
                product["short_description"], bedrock_runtime
            )
            print(f"Processed {i+1}/{len(products)} products")

            # Optional: Add small delay to avoid rate limits
            time.sleep(0.1)

        except Exception as e:
            print(f"Error processing product {i}: {e}")
            product["embedding"] = None

    with open("data/products_with_embeddings.json", "w") as f:
        json.dump(products, f, indent=2)
