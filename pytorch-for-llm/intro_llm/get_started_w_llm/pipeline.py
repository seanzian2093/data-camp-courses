""" Using Hugging face pipeline to summarize a long text """

from transformers import pipeline

# Load the model pipeline
summarizer = pipeline(task="summarization", model="cnicu/t5-small-booksum")

# Pass the long text to the model
output = summarizer(long_text, max_length=50)

# Access and print the summarized text
print(output[0]["summary_text"])
