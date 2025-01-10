""" A RNN for text generation, e.g., autocomplete, text completion, etc. """

import torch
import torch.nn as nn

# Prepare data

data = """The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, 
so suddenly that Alice had not a moment to think about stopping herself before she found herself falling 
down a very deep well.
"""

# `inputs` should have 218 rows which is the length of the data, excluding terminator character
# for demo purpose use 6
inputs = torch.tensor(
    [
        [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    ]
)

targets = torch.tensor(
    [
        1,
        3,
        2,
        5,
        4,
        0,
    ]
)
chars = [
    "r",
    "w",
    "T",
    "e",
    "a",
    "y",
    "m",
    "t",
    "k",
    "o",
    "h",
    "u",
    "n",
    "A",
    "g",
    "s",
    "f",
    "l",
    "p",
    ",",
    " ",
    "v",
    "d",
    ".",
    "c",
    "-",
    "b",
    "i",
]
char_to_ix = {
    "r": 0,
    "w": 1,
    "T": 2,
    "e": 3,
    "a": 4,
    "y": 5,
    "m": 6,
    "t": 7,
    "k": 8,
    "o": 9,
    "h": 10,
    "u": 11,
    "n": 12,
    "A": 13,
    "g": 14,
    "s": 15,
    "f": 16,
    "l": 17,
    "p": 18,
    ",": 19,
    " ": 20,
    "v": 21,
    "d": 22,
    ".": 23,
    "c": 24,
    "-": 25,
    "b": 26,
    "i": 27,
}

ix_to_char = {
    0: "r",
    1: "w",
    2: "T",
    3: "e",
    4: "a",
    5: "y",
    6: "m",
    7: "t",
    8: "k",
    9: "o",
    10: "h",
    11: "u",
    12: "n",
    13: "A",
    14: "g",
    15: "s",
    16: "f",
    17: "l",
    18: "p",
    19: ",",
    20: " ",
    21: "v",
    22: "d",
    23: ".",
    24: "c",
    25: "-",
    26: "b",
    27: "i",
}


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# Initiate the model
# model = RNNModel(input_size=len(data), hidden_size=16, output_size=len(data))
model = RNNModel(input_size=6, hidden_size=16, output_size=6)

# Initialize the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(100):
    model.train()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/100, Loss: {loss.item()}")

# Test the model
model.eval()
test_input = char_to_ix["r"]
test_input = nn.functional.one_hot(
    # torch.tensor(test_input).view(-1, 1), num_classes=len(chars)
    # reshapes the tensor to have one column and as many rows as needed to accommodate all elements
    # effectively converting the tensor into a column vector.
    torch.tensor(test_input).view(-1, 1),
    num_classes=6,
).float()
predicted_output = model(test_input)
predicted_char_ix = torch.argmax(predicted_output, 1).item()
print(f"Test Input: 'r', Predicted Output: '{ix_to_char[predicted_char_ix]}'")
