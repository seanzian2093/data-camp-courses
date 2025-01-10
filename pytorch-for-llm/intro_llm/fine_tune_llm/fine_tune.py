import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset


# Prepare the data
class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


train_data = CustomDataset(
    [
        {
            "interaction": "I'm really upset with the delays on delivering this item. Where is it?",
            "risk": "high risk",
        },
        {
            "interaction": "The support I've had on this issue has been terrible and really unhelpful. Why will no one help me?",
            "risk": "high risk",
        },
        {
            "interaction": "I have a question about how to use this product. Can you help me?",
            "risk": "low risk",
        },
        {
            "interaction": "This product is listed as out of stock. When will it be available again?",
            "risk": "low risk",
        },
    ]
)

test_data = CustomDataset(
    [
        {
            "interaction": "You charged me twice for the one item. I need a refund.",
            "risk": "high risk",
        }
    ]
)

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the data
tokenized_training_data = tokenizer(
    train_data["interaction"],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=20,
)

tokenized_test_data = tokenizer(
    test_data["interaction"],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=20,
)

print(tokenized_training_data)


# Tokenize row by row - more control over the tokenization process
def tokenize_function(data):
    return tokenizer(
        data["interaction"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )


tokenized_by_row = train_data.map(tokenize_function, batched=False)

print(tokenized_by_row)

# Set up an instance of TrainingArguments
training_args = TrainingArguments(
    output_dir="./finetuned",
    # Set the evaluation strategy
    evaluation_strategy="epoch",
    # Specify the number of epochs
    num_train_epochs=3,
    learning_rate=2e-5,
    # Set the batch sizes
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    weight_decay=0.01,
)

# Set up the trainer object
trainer = Trainer(
    model=model,
    # Assign the training arguments and tokenizer
    args=training_args,
    train_dataset=tokenized_training_data,
    eval_dataset=tokenized_test_data,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Use the fine-tuned model to make predictions
input_text = [
    "I'd just like to say, I love the product! Thank you!",
    "I'm really disappointed",
]

# Tokenize the new data
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# Pass the tokenized inputs through the model - no need to calculate gradients
with torch.no_grad():
    outputs = model(**inputs)

# Extract the new predictions
predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()

label_map = {0: "Low risk", 1: "High risk"}
for i, predicted_label in enumerate(predicted_labels):
    churn_label = label_map[predicted_label]
    print(f"\n Input Text {i + 1}: {input_text[i]}")
    print(f"Predicted Label: {churn_label}")
