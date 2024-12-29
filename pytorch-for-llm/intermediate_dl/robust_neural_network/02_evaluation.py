import importlib.util
import os
import sys

import torch
from torchmetrics import Accuracy
from torch.utils.data import DataLoader

# Path to the module
module_dir = "/Users/s0046425/git_projects/2025/data-camp-courses/pytorch-for-llm/intermediate_dl_pytorch"


# Function to load the module
def load_module(module_fp, module_name=None):
    spec = importlib.util.spec_from_file_location(module_name, module_fp)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Load the module - 00
module_fp_00 = os.path.join(module_dir, "00_dataset_dataloader.py")
module_name_00 = "dd"
dd = load_module(module_fp_00, module_name_00)

# Load the module - 01
module_fp_01 = os.path.join(module_dir, "01_model.py")
module_name_01 = "md"
md = load_module(module_fp_01, module_name_01)

dataset = dd.RandomDataset(num_samples=1000, num_features=10)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

acc = Accuracy(task="binary")

net = md.SimpleNeuralNetwork(
    input_size=10, hidden_size1=20, hidden_size2=10, output_size=1
)

net.eval()
with torch.no_grad():
    for data, labels in dataloader:
        outputs = net(data)
        preds = (outputs > 0.5).float()
        acc(preds, labels.view(-1, 1))

test_acc = acc.compute()
print(f"Test accuracy: {test_acc}")
