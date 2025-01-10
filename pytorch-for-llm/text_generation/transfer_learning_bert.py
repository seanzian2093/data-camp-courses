""" BERT, a pre-trained transformer model, is used for transfer learning. """

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Prepare the training data
texts = [
    "I love this!",
    "This is terrible.",
    "Amazing experience!",
    "Not my cup of tea.",
]
labels = [1, 0, 1, 0]

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Tokenize your data and return PyTorch tensors
inputs = tokenizer(
    texts, padding=True, truncation=True, return_tensors="pt", max_length=32
)
inputs["labels"] = torch.tensor(labels)

# Setup the optimizer using model parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
model.train()
for epoch in range(1):
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# Evaluate the model
text = "I had an awesome day!"

# Tokenize the text and return PyTorch tensors
input_eval = tokenizer(
    text, return_tensors="pt", truncation=True, padding=True, max_length=32
)
outputs_eval = model(**input_eval)

# Convert the output logits to probabilities
predictions = torch.nn.functional.softmax(outputs_eval.logits, dim=-1)

# Display the sentiments
predicted_label = "positive" if torch.argmax(predictions) > 0 else "negative"
print(f"Text: {text}\nSentiment: {predicted_label}")
