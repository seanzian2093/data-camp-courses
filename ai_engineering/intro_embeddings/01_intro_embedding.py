import json

# Cannot direct import due to the py file name starts with a number
utils = __import__("00_utils")
bedrock_runtime = utils.bedrock_runtime

# Text to be embedded
text = "This can contain any text."

# Try with Amazon Titan embeddings model
body = json.dumps({"inputText": text})

# Create a request to obtain embeddings - different model may requrire different parameters
response = bedrock_runtime.invoke_model(
    modelId="amazon.titan-embed-text-v1",  # Try this model instead
    body=body,
    contentType="application/json",
)

# Parse the response
response_body = json.loads(response["body"].read())
embedding = response_body["embedding"]

# Example usage
print(f"Embedding created successfully. Dimension: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
