""" Regard measures polarity in two lists of texts associated with two different classes. """

import evaluate

group1 = ["abc are described as loyal", "abc are honest but kind"]
group2 = ["abc are known for being confrontational", "abc are very blunt"]
# Load the regard and regard-comparison metrics
regard = evaluate.load("regard")
regard_comp = evaluate.load("regard", "compare")

# Compute the regard (polarities) of each group separately
polarity_results_1 = regard.compute(data=group1)
print("Polarity in group 1:\n", polarity_results_1)
polarity_results_2 = regard.compute(data=group2)
print("Polarity in group 2:\n", polarity_results_2)

# Compute the relative regard between the two groups for comparison
polarity_results_comp = regard_comp.compute(data=group1, references=group2)
print("Polarity comparison between groups:\n", polarity_results_comp)
