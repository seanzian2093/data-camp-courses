"""
Rouge measures the overlap between the generated summary and the reference summary, looking at n-grams and overlapping word sequences.

Meteor focuses on the semantic similarity between the generated summary and the reference summary, i.e.,
word variation, word order, and word semantics. It is a metric that is based on the harmonic mean of unigram precision and recall.

Exact-match is a metric that measures the percentage of generated summaries that are EXACTLY the same as the reference summaries.

"""

import evaluate

# Load the rouge metric
rouge = evaluate.load("rouge")

predictions = [
    """Pluto is a dwarf planet in our solar system, located in the Kuiper Belt beyond Neptune, and was formerly considered the ninth planet until its reclassification in 2006."""
]
references = [
    """Pluto is a dwarf planet in the solar system, located in the Kuiper Belt beyond Neptune, and was previously deemed as a planet until it was reclassified in 2006."""
]

# Calculate the rouge scores between the predicted and reference summaries
results = rouge.compute(predictions=predictions, references=references)
print("ROUGE results: ", results)

meteor = evaluate.load("meteor")

generated = [
    "The burrow stretched forward like a narrow corridor for a while, then plunged abruptly downward, so quickly that Alice had no chance to stop herself before she was tumbling into an extremely deep shaft."
]
reference = [
    "The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well."
]

# Compute and print the METEOR score
results = meteor.compute(predictions=generated, references=reference)
print("Meteor: ", results["meteor"])

# Load the metric
exact_match = evaluate.load("exact_match")

predictions = [
    "It's a wonderful day",
    "I love dogs",
    "DataCamp has great AI courses",
    "Sunshine and flowers",
]
references = [
    "What a wonderful day",
    "I love cats",
    "DataCamp has great AI courses",
    "Sunsets and flowers",
]

# Compute the exact match and print the results
results = exact_match.compute(references=references, predictions=predictions)
print("EM results: ", results)
