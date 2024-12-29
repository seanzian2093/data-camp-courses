from torchtext.data.utils import get_tokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


text = """In the city of Dataville, a data analyst named Alex explores hidden insights within vast data.
With determination, Alex uncovers patterns, cleanses the data, and unlocks innovation. Join this 
adventure to unleash the power of data-driven decisions.
"""

# Initialize the tokenizer
tokenizer = get_tokenizer("basic_english")

# Tokenize the text
tokens = tokenizer(text)

# Remove any stopwords
stop_words = set(stopwords.words("english"))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

# Perform stemming on the filtered tokens
stemmer = PorterStemmer()
stemmed_token = [stemmer.stem(token) for token in filtered_tokens]
print(stemmed_token)
