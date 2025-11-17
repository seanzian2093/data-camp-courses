import scipy.spatial.distance as distance

bedrock_runtime = __import__("00_utils").bedrock_runtime
create_embedding = __import__("02_embed_description").create_embedding

def create_embeddings(texts):
    embeddings = [create_embedding(text, bedrock_runtime) for text in texts]
    return embeddings

sentiments = [{'label': 'Positive'},
              {'label': 'Neutral'},
              {'label': 'Negative'}]

reviews = ["The food was delicious!",
           "The service was a bit slow but the food was good",
           "The food was cold, really disappointing!"]

sentiments_2 = [{'label': 'Positive',
               'description': 'A positive restaurant review'},
              {'label': 'Neutral',
               'description':'A neutral restaurant review'},
              {'label': 'Negative',
               'description': 'A negative restaurant review'}]

# Create a list of class descriptions from the sentiment labels
class_descriptions = [sentiment['label'] for sentiment in sentiments]
class_descriptions_2 = [sentiment['description'] for sentiment in sentiments_2]

# Embed the class_descriptions and reviews
class_embeddings = create_embeddings(class_descriptions)
class_embeddings_2 = create_embeddings(class_descriptions_2)
review_embeddings = create_embeddings(reviews)

# Define a function to return the minimum distance and its index
def find_closest(query_vector, embeddings):
    distances = []
    for index, embedding in enumerate(embeddings):
        dist = distance.cosine(query_vector, embedding)
        distances.append({"distance": dist, "index": index})
    return min(distances, key=lambda x: x["distance"])

print("\nUsing label for classification:")
for index, review in enumerate(reviews):
    # Find the closest distance and its index using find_closest()
    closest = find_closest(review_embeddings[index], class_embeddings)
    # Subset sentiments using the index from closest
    label = sentiments[closest['index']]['label']
    print(f'"{review}" was classified as {label}')

print("\nUsing descriptions for classification:")
for index, review in enumerate(reviews):
    closest = find_closest(review_embeddings[index], class_embeddings_2)
    label = sentiments_2[closest['index']]['label']
    print(f'"{review}" was classified as {label}')