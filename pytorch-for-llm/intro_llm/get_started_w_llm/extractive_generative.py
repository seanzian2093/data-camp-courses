"""
Question-answering can be either extractive or generative, each requiring a different transformer structure to process input and output correctly.
Extractive models are used to extract the answer from the input text, while generative models are used to generate the answer from scratch.
Extractive models are typically encoder-only models, while generative models are decoder-only models.

Encoder-only BERT-based models such as "distilbert-base-uncased-distilled-squad"
Decoder-only models such as "gpt2"
"""

from transformers import pipeline

# Prepare the text
text = "\nThe Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci. Considered an archetypal masterpiece of the Italian Renaissance, it has been described as the most known, visited, talked about, and sung about work of art in the world. The painting's novel qualities include the subject's enigmatic expression, the monumentality of the composition, and the subtle modeling of forms.\n"

# Ask the question
question = "Who painted the Mona Lisa?"

# Define the appropriate model
qa = pipeline(
    task="question-answering", model="distilbert-base-uncased-distilled-squad"
)

output = qa(question=question, context=text)
print(output["answer"])

question = "Who painted the Mona Lisa?"

# Define the appropriate model
qa = pipeline(task="question-answering", model="gpt2")

output = qa({"context": text, "question": question}, max_length=150)
print(output["answer"])
