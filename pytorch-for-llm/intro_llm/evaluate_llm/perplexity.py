""" 
Perplexity evaluates a language model's ability to predict next word accurately and confidently. 
Lower perplexity indicates higher confidence, ie., better performance. 

In short, perplexity for text generation.
"""

import evaluate
from transformers import AutoModelForCausalLM, GPT2TokenizerFast

# Prepare model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Define the input text
input_text = "Current trends show that by 2030"

# Encode the input text, generate and decode it
input_text_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_text_ids, max_length=20)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text: ", generated_text)

# Load and compute the perplexity score
perplexity = evaluate.load("perplexity", module_type="metric")
results = perplexity.compute(model_id="gpt2", predictions=generated_text)
print("Perplexity: ", results["mean_perplexity"])
