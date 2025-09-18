from utils import spark
from pyspark.sql.functions import round, lit
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)

# Read data from CSV file
flights = spark.read.csv(
    "data/flights.csv", sep=",", header=True, inferSchema=True, nullValue="NA"
)

# Remove the 'flight' column
flights_drop_column = flights.drop("flight")

# Number of records with missing 'delay' values
flights_drop_column.filter("delay IS NULL").count()

# Remove records with missing 'delay' values
flights_valid_delay = flights_drop_column.filter("delay IS NOT NULL")

# Remove records with missing values in any column and get the number of remaining rows
flights_none_missing = flights_valid_delay.dropna()
print(flights_none_missing.count())

flights = flights_none_missing
flights.printSchema()
# Convert 'mile' to 'km' and drop 'mile' column (1 mile is equivalent to 1.60934 km)
# flights_km = flights.withColumn("km", round(flights.mile * 1.60934, 0)).drop("mile")
flights_km = flights.withColumn("km", round(flights.mile * lit(1.60934), 0)).drop(
    "mile"
)

# Create 'label' column indicating whether flight delayed (1) or not (0)
# flights_km = flights_km.withColumn("label", (flights_km.delay >= 15).cast("integer"))
flights_km = flights_km.withColumn(
    "label", (flights_km.delay >= lit(15)).cast("integer")
)

# Check first five records
flights_km.show(5)


flights = flights_km
# Create an indexer
indexer = StringIndexer(inputCol="carrier", outputCol="carrier_idx")

# Indexer identifies categories in the data
indexer_model = indexer.fit(flights)

# Indexer creates a new column with numeric index values
flights_indexed = indexer_model.transform(flights)

# Repeat the process for the other categorical feature
flights_indexed = (
    StringIndexer(inputCol="org", outputCol="org_idx")
    .fit(flights_indexed)
    .transform(flights_indexed)
)
flights_indexed.show(5)

flights = flights_indexed
# Create an assembler object
assembler = VectorAssembler(
    inputCols=[
        "mon",
        "dom",
        "dow",
        "carrier_idx",
        "org_idx",
        "km",
        "depart",
        "duration",
    ],
    outputCol="features",
)

# Consolidate predictor columns
flights_assembled = assembler.transform(flights)

# Check the resulting column
flights_assembled.select("features", "delay").show(5, truncate=False)

flights = flights_assembled
# Split into training and testing sets in a 80:20 ratio
flights_train, flights_test = flights.randomSplit([0.8, 0.2], 43)

# Check that training set has around 80% of records
training_ratio = flights_train.count() / flights.count()
print(training_ratio)


# Create a classifier object and fit to the training data
tree = DecisionTreeClassifier()
tree_model = tree.fit(flights_train)

# Create predictions for the testing data and take a look at the predictions
prediction = tree_model.transform(flights_test)
prediction.select("label", "prediction", "probability").show(5, False)

# Create a confusion matrix
prediction.groupBy("label", "prediction").count().show()

# Calculate the elements of the confusion matrix
TN = prediction.filter("prediction = 0 AND label = prediction").count()
TP = prediction.filter("prediction = 1 AND label = prediction").count()
FN = prediction.filter("prediction = 0 AND label != prediction").count()
FP = prediction.filter("prediction = 1 AND label != prediction").count()

# Accuracy measures the proportion of correct predictions
accuracy = (TN + TP) / (TN + TP + FN + FP)
print(accuracy)

# Create a classifier object and train on training data
logistic = LogisticRegression().fit(flights_train)

# Create predictions for the testing data and show confusion matrix
prediction = logistic.transform(flights_test)
prediction.groupBy("label", "prediction").count().show()

# Calculate precision and recall - all about true positives
precision = TP / (TP + FP)
recall = TP / (TP + FN)
print("precision = {:.2f}\nrecall    = {:.2f}".format(precision, recall))

# Find weighted precision
multi_evaluator = MulticlassClassificationEvaluator()
weighted_precision = multi_evaluator.evaluate(
    prediction, {multi_evaluator.metricName: "weightedPrecision"}
)
print(f"weighted precision = {weighted_precision}")

# Find AUC
binary_evaluator = BinaryClassificationEvaluator()
auc = binary_evaluator.evaluate(
    prediction, {binary_evaluator.metricName: "areaUnderROC"}
)
print(f"AUC = {auc}")
