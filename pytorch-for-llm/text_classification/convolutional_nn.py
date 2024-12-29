import torch.nn as nn
import torch.nn.functional as F


class TextClassificationCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TextClassificationCNN, self).__init__()

        # Initialize the embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )

        # Convolutional layer
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.fc = nn.Linear(embedding_dim, 2)

    def forward(self, text):
        # changes the shape of the tensor from - sequence_length is the length of the input text, vocab size
        # (batch_size, sequence_length, embedding_dim) to
        # (batch_size, embedding_dim, sequence_length).
        embedded = self.embedding(text).permute(0, 2, 1)
        conved = F.relu(self.conv(embedded))
        # The mean is taken over the sequence length dimension, ie the 3rd dimension.
        conved = conved.mean(dim=2)
        return self.fc(conved)
