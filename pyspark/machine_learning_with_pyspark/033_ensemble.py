from utils import spark
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Read data from CSV file
flights = spark.read.csv(
    "data/flights.csv", sep=",", header=True, inferSchema=True, nullValue="NA"
)

# Remove the 'flight' column of null values
flights = flights.filter("delay IS NOT NULL")

# Create label column indicating whether flight delayed (1) or not (0)
flights = flights.withColumn("label", (flights.delay >= 15).cast("integer"))

# Assemble predictors into a single column
assembler = VectorAssembler(
    inputCols=["mon", "depart", "duration"], outputCol="features"
)
flights = assembler.transform(flights)

# Split data into training and testing sets
flights_train, flights_test = flights.randomSplit([0.8, 0.2], seed=42)


# Create model objects and train on training data
tree = DecisionTreeClassifier().fit(flights_train)
gbt = GBTClassifier().fit(flights_train)

# Compare AUC on testing data
evaluator = BinaryClassificationEvaluator()
print(evaluator.evaluate(tree.transform(flights_test)))
print(evaluator.evaluate(gbt.transform(flights_test)))

# Find the number of trees and the relative importance of features
print(gbt.getNumTrees)
print(gbt.featureImportances)
