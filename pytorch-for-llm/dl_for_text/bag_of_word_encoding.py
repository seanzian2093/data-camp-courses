"""Bag of words encoding is a simple and effective way to represent text data. """

from sklearn.feature_extraction.text import CountVectorizer

titles = [
    "The Great Gatsby",
    "To Kill a Mockingbird",
    "1984",
    "The Catcher in the Rye",
    "The Hobbit",
    "Great Expectations",
]

# Initialize the count vectorizer
vectorizer = CountVectorizer()
bow_encoded_titles = vectorizer.fit_transform(titles)

# Extract and print features
print(vectorizer.get_feature_names_out())
print(bow_encoded_titles.toarray())
