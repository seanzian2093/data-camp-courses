from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
from utils import spark

# Read data from CSV file
flights = spark.read.csv(
    "data/flights.csv", sep=",", header=True, inferSchema=True, nullValue="NA"
)

# mile to km
# flights = flights.withColumn("km", round(flights.mile * 1.60934, 0)).drop("mile")
flights = flights.withColumn("km", flights.mile * 1.60934).drop("mile")

# Split data into training and testing sets
flights_train, flights_test = flights.randomSplit([0.8, 0.2], seed=42)

# Create parameter grid
params = ParamGridBuilder()

# Create objects for building and evaluating a regression model
regression = LinearRegression(labelCol="duration")
evaluator = RegressionEvaluator(labelCol="duration")

# Create an indexer for the org field
indexer = StringIndexer(inputCol="org", outputCol="org_idx")

# Create an one-hot encoder for the indexed org field
onehot = OneHotEncoder(inputCol="org_idx", outputCol="org_dummy")

# Assemble the km and one-hot encoded fields
assembler = VectorAssembler(inputCols=["km", "org_dummy"], outputCol="features")

# Create a pipeline.
pipeline = Pipeline(stages=[indexer, onehot, assembler, regression])

# Add grids for two parameters
params = params.addGrid(regression.regParam, [0.01, 0.1, 1.0, 10.0]).addGrid(
    regression.elasticNetParam, [0.0, 0.5, 1.0]
)

# Build the parameter grid
params = params.build()
print("Number of models to be tested: ", len(params))

# Create cross-validator.
cv = CrossValidator(
    estimator=pipeline, estimatorParamMaps=params, evaluator=evaluator, numFolds=5
)

# Train and test model on multiple folds of the training data
cv = cv.fit(flights_train)

# Get the best model from cross validation
best_model = cv.bestModel

# Look at the stages in the best model
print(best_model.stages)

# Get the parameters for the LinearRegression object in the best model
best_model.stages[3].extractParamMap()

# Generate predictions on testing data using the best model then calculate RMSE
predictions = best_model.transform(flights_test)
print("RMSE =", evaluator.evaluate(predictions))

# Average AUC for each parameter combination in grid
print(cv.avgMetrics)

# Average AUC for the best model
print(max(cv.avgMetrics))

# What's the optimal parameter value for regParam?
print(cv.bestModel.explainParams())
# What's the optimal parameter value for featureSubsetStrategy?
# print(cv.bestModel.explainParam("elasticNetParam"))
