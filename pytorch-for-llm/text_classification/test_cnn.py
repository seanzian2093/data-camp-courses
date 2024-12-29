import torch
from train_cnn import word_to_ix, model

book_reviews = [
    "I love this book".split(),
    "I do not like this book".split(),
]

for review in book_reviews:
    input_tensor = torch.LongTensor([word_to_ix.get(word, 0) for word in review])
    print(input_tensor.shape)
    input_tensor = input_tensor.unsqueeze(0)
    print(input_tensor.shape)
    # input_tensor = torch.tensor(
    #     [word_to_ix.get(word, 0) for word in review], dtype=torch.long
    # ).unsqueeze(0)

    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)

    sentiment = "positive" if predicted.item() == 1 else "negative"
    print(f"Review: {' '.join(review)} Sentiment: {sentiment}")
