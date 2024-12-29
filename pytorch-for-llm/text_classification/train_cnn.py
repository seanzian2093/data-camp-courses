import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from convolutional_nn import TextClassificationCNN as net

# Define the model - vocab_size is the number of unique words in the vocabulary/data
model = net(vocab_size=17, embedding_dim=10)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Define the data
data = [
    (["I", "love", "this", "book"], 1),
    (["This", "is", "an", "amazing", "novel"], 1),
    (["I", "really", "like", "this", "story"], 1),
    (["I", "do", "not", "like", "this", "book"], 0),
    (["I", "hate", "this", "novel"], 0),
    (["This", "is", "a", "terrible", "story"], 0),
]

word_to_ix = {
    "I": 0,
    "love": 1,
    "this": 2,
    "book": 3,
    "This": 4,
    "is": 5,
    "an": 6,
    "amazing": 7,
    "novel": 8,
    "really": 9,
    "like": 10,
    "story": 11,
    "do": 12,
    "not": 13,
    "hate": 14,
    "a": 15,
    "terrible": 16,
}
# Training loop

for epoch in range(10):
    for sentence, label in data:
        model.zero_grad()
        # `.unsqueeze(0)` adds a batch dimension
        sentence = torch.LongTensor(
            [word_to_ix.get(word, 0) for word in sentence]
        ).unsqueeze(0)

        label = torch.LongTensor([int(label)])
        outputs = model(sentence)
        loss = criterion(outputs, label)
        loss.backward()
        # Update the weights/parameters
        optimizer.step()
print("Training complete")
