from torchtext.data.utils import get_tokenizer
from nltk.probability import FreqDist

text = """In the city of Dataville, a data analyst named Alex explores hidden insights within vast data.
With determination, Alex uncovers patterns, cleanses the data, and unlocks innovation. Join this 
adventure to unleash the power of data-driven decisions.
"""

# Initialize the tokenizer
tokenizer = get_tokenizer("basic_english")

# Tokenize the text
tokens = tokenizer(text)

# Remove rare words
threshold = 1
freq_dist = FreqDist(tokens)
common_tokens = [token for token in tokens if freq_dist[token] > threshold]
print(common_tokens)
