import torch
import torch.nn as nn

# Previous model
# RNNModel(
#   (embeddings): Embedding(17, 10)
#   (rnn): RNN(10, 16, batch_first=True)
#   (fc): Linear(in_features=16, out_features=17, bias=True)
# )

# Define the model
vocab_size = 17
embedding_dim = 10
hidden_dim = 16


class RNNWithAttentionModel(nn.Module):
    def __init__(self):
        super(RNNWithAttentionModel, self).__init__()
        # Create an embedding layer for the vocabulary
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        # Apply a linear transformation to get the attention scores
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)
        out, _ = self.rnn(x)
        #  Get the attention weights
        attn_weights = torch.nn.functional.softmax(
            self.attention(out).squeeze(2), dim=1
        )
        # Compute the context vector
        context = torch.sum(attn_weights.unsqueeze(2) * out, dim=1)
        out = self.fc(context)
        return out


attention_model = RNNWithAttentionModel()
optimizer = torch.optim.Adam(attention_model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
print("Model Instantiated")

# Train the model

inputs = [
    torch.tensor([9, 15, 0, 3, 9]),
    torch.tensor([12, 1, 5, 7]),
    torch.tensor([2, 1, 14, 13]),
    torch.tensor([4, 1, 9, 8]),
]
targets = torch.tensor([10, 6, 11, 16])

input_data = [[9, 15, 0, 3, 9], [12, 1, 5, 7], [2, 1, 14, 13], [4, 1, 9, 8]]
target_data = [10, 6, 11, 16]

ix_to_word = {
    0: "sat",
    1: "are",
    2: "parrots",
    3: "on",
    4: "whales",
    5: "very",
    6: "animals",
    7: "loyal",
    8: "largest",
    9: "the",
    10: "mat",
    11: "noisy",
    12: "dogs",
    13: "and",
    14: "colorful",
    15: "cat",
    16: "mammals",
}


def pad_sequences(batch):
    max_len = max([len(seq) for seq in batch])
    return torch.stack(
        [torch.cat([seq, torch.zeros(max_len - len(seq)).long()]) for seq in batch]
    )


for epoch in range(300):
    attention_model.train()
    optimizer.zero_grad()
    padded_inputs = pad_sequences(inputs)
    outputs = attention_model(padded_inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

for input_seq, target in zip(input_data, target_data):
    input_test = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0)

    #  Set the RNN model to evaluation mode
    rnn_model.eval()
    # Get the RNN output by passing the appropriate input
    rnn_output = rnn_model(input_test)
    # Extract the word with the highest prediction score
    rnn_prediction = ix_to_word[torch.argmax(rnn_output).item()]

    attention_model.eval()
    attention_output = attention_model(input_test)
    # Extract the word with the highest prediction score
    attention_prediction = ix_to_word[torch.argmax(attention_output).item()]

    print(f"\nInput: {' '.join([ix_to_word[ix] for ix in input_seq])}")
    print(f"Target: {ix_to_word[target]}")
    print(f"RNN prediction: {rnn_prediction}")
    print(f"RNN with Attention prediction: {attention_prediction}")
