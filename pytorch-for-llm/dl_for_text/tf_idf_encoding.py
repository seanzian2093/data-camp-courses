""" Use TF-IDF encoding to convert text data into numerical data to identify important words. """

from sklearn.feature_extraction.text import TfidfVectorizer

titles = [
    "The Great Gatsby",
    "To Kill a Mockingbird",
    "1984",
    "The Catcher in the Rye",
    "The Hobbit",
    "Great Expectations",
]

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_encoded_titles = vectorizer.fit_transform(titles)

# Extract and print features
print(vectorizer.get_feature_names_out())
print(tfidf_encoded_titles.toarray())
