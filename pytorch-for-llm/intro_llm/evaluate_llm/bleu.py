"""
Bleu measures translation quality against human references.
In short, bleu for text generation/summarization/question-answering.
"""

import evaluate
from transformers import pipeline

# Load the blue metric
bleu = evaluate.load("bleu")

# Prepare inputs and references
input_sentence_1 = "Hola, ¿cómo estás?"

reference_1 = [["Hello, how are you?", "Hi, how are you?"]]

input_sentences_2 = ["Hola, ¿cómo estás?", "Estoy genial, gracias."]

references_2 = [
    ["Hello, how are you?", "Hi, how are you?"],
    ["I'm great, thanks.", "I'm great, thank you."],
]

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

# Translate the first input sentence then calucate the BLEU metric for translation quality
translated_output = translator(input_sentence_1)

translated_sentence = translated_output[0]["translation_text"]

print("Translated:", translated_sentence)

results = bleu.compute(predictions=[translated_sentence], references=reference_1)
print(results)
