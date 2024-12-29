from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Input text
processed_text = [
    "project gutenberg ebook complet work william shakespear , william shakespear ebook use anyon anywher unit state part world cost almost restrict whatsoev .",
    "may copi , give away re-us term project gutenberg licens includ ebook onlin www .",
    "locat unit state , check law countri locat use ebook .",
    "titl complet work william shakespear author william shakespear releas date januari 1994 [ebook #100] [most recent updat may 18 , 2023] languag english *** start project gutenberg ebook complet work william shakespear *** complet work william shakespear william shakespear content sonnet all’ well end well tragedi antoni cleopatra like comedi error tragedi coriolanu cymbelin tragedi hamlet , princ denmark first part king henri fourth second part king henri fourth life king henri fifth first part henri sixth second part king henri sixth third part king henri sixth king henri eighth life death king john tragedi juliu caesar tragedi king lear love’ labour’ lost tragedi macbeth measur measur merchant venic merri wive windsor midsumm night’ dream much ado noth tragedi othello , moor venic pericl , princ tyre king richard second king richard third tragedi romeo juliet tame shrew tempest life timon athen tragedi titu andronicu troilu cressida twelfth night , two gentlemen verona two nobl kinsmen winter’ tale lover’ complaint passion pilgrim phoenix turtl rape lucrec venu adoni sonnet 1 fairest creatur desir increas , therebi beauty’ rose might never die , riper time deceas , tender heir might bear memori thou contract thine bright eye , feed’st thi light’ flame self-substanti fuel , make famin abund lie , thyself thi foe , thi sweet self cruel thou art world’ fresh ornament , herald gaudi spring , within thine bud buriest thi content , , tender churl , mak’st wast niggard piti world , els glutton , eat world’ due , grave thee .",
    "forti winter shall besieg thi brow , dig deep trench thi beauty’ field , thi youth’ proud liveri gaze , tatter weed small worth held ask , thi beauti lie , treasur thi lusti day say , within thine deep sunken eye , all-eat shame , thriftless prais .",
    "much prais deserv’d thi beauty’ use , thou couldst answer ‘thi fair child mine shall sum count , make old excus , ’ prove beauti success thine .",
    "new made thou art old , see thi blood warm thou feel’st cold .",
    "look thi glass tell face thou viewest , time face form anoth , whose fresh repair thou renewest , thou dost beguil world , unbless mother .",
    "fair whose unear womb disdain tillag thi husbandri ?",
]

# Create a list of stopwords
stop_words = set(stopwords.words("english"))

# Initialize the tokenizer and stemmer
tokenizer = get_tokenizer("basic_english")
stemmer = PorterStemmer()


# Define a preprocessing pipeline function
def preprocess_sentenses(sentences):
    preprocessed_sentences = []
    for sentence in sentences:
        # Tokenize the text
        tokens = tokenizer(sentence)
        # Remove any stopwords
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
        # Perform stemming on the filtered tokens
        stemmed_token = [stemmer.stem(token) for token in filtered_tokens]
        preprocessed_sentences.append(stemmed_token)
    return preprocessed_sentences


# Define a Dataset class
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Define a encoding function
def encode_sentence(sentences):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    return X.toarray(), vectorizer


# Define a text processing pipeline
def text_processing_pipeline(sentences):
    preprocessed_sentences = preprocess_sentenses(sentences)
    encoded_sentences, vectorizer = encode_sentence(preprocessed_sentences)
    dataset = TextDataset(encoded_sentences)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader, vectorizer


dataloader, vectorizer = text_processing_pipeline(processed_text)
print(vectorizer.get_feature_names_out()[:10])
print(next(iter(dataloader))[0][:10])
