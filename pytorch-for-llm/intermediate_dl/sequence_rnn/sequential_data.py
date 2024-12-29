import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


# Assume df is a pandas dataframe with two columns: 'index' and 'value'
# from `i`th row to `i + seq_length`th row, exclusively, the value of 'value' is the input, ie. x
# and the value of 'value' in `i + seq_length`th row is the target, ie, y
def create_sequences(df, seq_length):
    xs, ys = [], []
    # len(df) returns the number of rows in the dataframe
    for i in range(len(df) - seq_length):
        x = df.iloc[i : (i + seq_length), 1]
        y = df.iloc[i + seq_length, 1]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


# Create a sample DataFrame
data = {
    "index": range(10016),
    # Generate 10016 random numbers - make sure last batch is of size 32
    "value": np.random.randn(10016),
}
train_data = pd.DataFrame(data)
print(train_data.head())

# Assume we have a dataframe `train_data` with two columns: 'index' and 'value'
X_train, y_train = create_sequences(train_data, seq_length=24 * 4)
print(X_train.shape, y_train.shape)

# Creaate TensorDataset
dataset_train = TensorDataset(
    torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
)
print(len(dataset_train))

dataset_test = dataset_train
