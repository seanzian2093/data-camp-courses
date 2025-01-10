"""
Use `evaluate` library to evaluate a LLM model. 
classic metrics are for text classification tasks.
perplexity is for text generation tasks.
bleu is for text generation/summarization/question-answering tasks.
rouge is for text summarization/question-answering tasks.
meteor is for translation tasks.
f1/exact-match is for question-answering tasks.
"""

import torch
import evaluate

# Load the metrics
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

# Obtain a description of each metric
print(accuracy.description)
print(precision.description)
print(recall.description)
print(f1.description)

# See the required data types
print(f"The required data types for accuracy are: {accuracy.features}.")
print(f"The required data types for precision are: {precision.features}.")
print(f"The required data types for recall are: {recall.features}.")
print(f"The required data types for f1 are: {f1.features}.")

# Create a dummy output - this might be the correct way to construct output of a model
outputs = torch.SequenceClassifierOutput(
    loss=None,
    logits=torch.tensor(
        [
            [-0.3532, 0.3247],
            [-0.2962, 0.2851],
            [-0.2947, 0.3434],
            [-0.3476, 0.3776],
            [-0.3089, 0.3227],
        ]
    ),
    hidden_states=None,
    attentions=None,
)

validate_labels = [0, 0, 1, 1, 1]

# Extract the new predictions
predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()

# Compute the metrics by comparing real and predicted labels
print(accuracy.compute(references=validate_labels, predictions=predicted_labels))
print(precision.compute(references=validate_labels, predictions=predicted_labels))
print(recall.compute(references=validate_labels, predictions=predicted_labels))
print(f1.compute(references=validate_labels, predictions=predicted_labels))
