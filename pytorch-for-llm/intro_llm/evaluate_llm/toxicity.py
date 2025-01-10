import evaluate

toxicity_metric = evaluate.load("toxicity", module_type="metric")
user_1 = ["Everyone that tried it love it", "This artist is a true genius, pure talent"]
user_2 = ["Nobody i've talked to likes this product", "Terrible singer"]

# Calculate the individual toxicities
toxicity_1 = toxicity_metric.compute(predictions=user_1)
toxicity_2 = toxicity_metric.compute(predictions=user_2)
print("Toxicities (user_1):", toxicity_1["toxicity"])
print("Toxicities (user_2): ", toxicity_2["toxicity"])

# Calculate the maximum toxicities
toxicity_1_max = toxicity_metric.compute(predictions=user_1, aggregation="maximum")
toxicity_2_max = toxicity_metric.compute(predictions=user_2, aggregation="maximum")
print("Maximum toxicity (user_1):", toxicity_1_max["max_toxicity"])
print("Maximum toxicity (user_2): ", toxicity_2_max["max_toxicity"])

# Calculate the toxicity ratios
toxicity_1_ratio = toxicity_metric.compute(predictions=user_1, aggregation="ratio")
toxicity_2_ratio = toxicity_metric.compute(predictions=user_2, aggregation="ratio")
print("Toxicity ratio (user_1):", toxicity_1_ratio["toxicity_ratio"])
print("Toxicity ratio (user_2): ", toxicity_2_ratio["toxicity_ratio"])
