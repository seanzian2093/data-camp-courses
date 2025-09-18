from utils import spark
from pyspark.sql.functions import round
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# Read data from CSV file
flights = spark.read.csv(
    "data/flights.csv", sep=",", header=True, inferSchema=True, nullValue="NA"
)

# mile to km
flights = flights.withColumn("km", round(flights.mile * 1.60934, 0)).drop("mile")

# Split data into training and testing sets
flights_train, flights_test = flights.randomSplit([0.8, 0.2], seed=42)

# Convert categorical strings to index values
indexer = StringIndexer(inputCols=["org"], outputCols=["org_idx"])

# One-hot encode index values
onehot = OneHotEncoder(
    inputCols=["org_idx", "dow"], outputCols=["org_dummy", "dow_dummy"]
)

# Assemble predictors into a single column
assembler = VectorAssembler(
    inputCols=["km", "org_dummy", "dow_dummy"], outputCol="features"
)

# A linear regression object
regression = LinearRegression(labelCol="duration")

# Import class for creating a pipeline

# Construct a pipeline
pipeline = Pipeline(stages=[indexer, onehot, assembler, regression])

# Train the pipeline on the training data
pipeline = pipeline.fit(flights_train)

# Make predictions on the testing data
predictions = pipeline.transform(flights_test)

# Calculate the RMSE
rmse = RegressionEvaluator(labelCol="duration", predictionCol="prediction").evaluate(
    predictions
)
print(f"test RMSE: {rmse}")
