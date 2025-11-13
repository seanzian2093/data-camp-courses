# Cannot direct import due to the py file name starts with a number
utils = __import__("00_utils")
client = utils.client

# Create a request to obtain embeddings
response = client.embeddings.create(
    model="text-embedding-3-small", input="This can contain any text."
)

# Convert the response into a dictionary
response_dict = response.model_dump()
print(response_dict)
