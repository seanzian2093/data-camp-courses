from utils import spark
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import regexp_replace
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)

# Read data from CSV file
schema = StructType(
    [
        StructField("id", IntegerType(), True),
        StructField("text", StringType(), True),
        StructField("label", IntegerType(), True),
    ]
)
sms = spark.read.csv("data/sms.csv", sep=";", header=False, schema=schema)

# Remove punctuation (REGEX provided) and numbers
wrangled = sms.withColumn("text", regexp_replace(sms.text, "[_():;,.!?\\-]", " "))
wrangled = wrangled.withColumn("text", regexp_replace(wrangled.text, "[\\d]", " "))

# Merge multiple spaces
wrangled = wrangled.withColumn("text", regexp_replace(wrangled.text, " +", " "))
sms_train, sms_test = wrangled.randomSplit([0.8, 0.2], 13)

# Break text into tokens at non-word characters
tokenizer = Tokenizer(inputCol="text", outputCol="words")

# Remove stop words
remover = StopWordsRemover(inputCol="words", outputCol="terms")

# Apply the hashing trick and transform to TF-IDF
hasher = HashingTF(inputCol="terms", outputCol="hash")
idf = IDF(inputCol="hash", outputCol="features")

# Create a logistic regression object and add everything to a pipeline
logistic = LogisticRegression()

# Create parameter grid
params = ParamGridBuilder()

# Add grid for hashing trick parameters
params = params.addGrid(hasher.numFeatures, [1024, 4096, 16384]).addGrid(
    hasher.binary, [True, False]
)

# Add grid for logistic regression parameters
params = params.addGrid(logistic.regParam, [0.01, 0.1, 1.0, 10.0]).addGrid(
    logistic.elasticNetParam, [0.0, 0.5, 1.0]
)

# Build parameter grid
params = params.build()
print("Number of models to be tested: ", len(params))

pipeline = Pipeline(stages=[tokenizer, remover, hasher, idf, logistic])
evaluator = BinaryClassificationEvaluator()

# Create cross-validator.
cv = CrossValidator(
    estimator=pipeline, estimatorParamMaps=params, evaluator=evaluator, numFolds=5
)

# Train and test model on multiple folds of the training data
cv = cv.fit(sms_train)

# Get the best model from cross validation
best_model = cv.bestModel

# Look at the stages in the best model
print(best_model.stages)

# Get the parameters for the LinearRegression object in the best model
best_model.stages[4].extractParamMap()

# Generate predictions on testing data using the best model then calculate RMSE
predictions = best_model.transform(sms_test)
print("RMSE =", evaluator.evaluate(predictions))

predictions.groupBy("label", "prediction").count().show()
