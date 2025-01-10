from transformers import pipeline

model = pipeline(
    task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Include an example in the input ext
input_text = """
Text: "The dinner we had was great and the service too."
Classify the sentiment of this sentence as either positive or negative.
Example:
Text: "The food was delicious"
Sentiment: Positive
Text: "The dinner we had was great and the service too."
Sentiment:
"""

# Apply the example to the model
result = model(input_text, max_length=100)

print(result[0]["label"])
