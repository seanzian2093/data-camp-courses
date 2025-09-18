from utils import spark
from pyspark.sql.functions import round
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer, Bucketizer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Read data from CSV file
flights = spark.read.csv(
    "data/flights.csv", sep=",", header=True, inferSchema=True, nullValue="NA"
)

# mile to km
flights = flights.withColumn("km", round(flights.mile * 1.60934, 0)).drop("mile")

# original airport
str_indexer = StringIndexer(inputCol="org", outputCol="org_idx").fit(flights)
flights = str_indexer.transform(flights)
# Create an instance of the one hot encoder
# onehot = OneHotEncoder(inputCols=["org_idx"], outputCols=["org_dummy"])
onehot = OneHotEncoder(inputCol="org_idx", outputCol="org_dummy")

# Apply the one hot encoder to the flights data
onehot = onehot.fit(flights)
flights = onehot.transform(flights)

# Check the results
flights.select("org", "org_idx", "org_dummy").distinct().orderBy("org_idx").show()

# departure time
# Create buckets at 3 hour intervals through the day
buckets = Bucketizer(
    splits=[0, 3, 6, 9, 12, 15, 18, 21, 24],
    inputCol="depart",
    outputCol="depart_bucket",
)

# Bucket the departure times
flights = buckets.transform(flights)
flights.select("depart", "depart_bucket").show(5)

# Create a one-hot encoder
onehot = OneHotEncoder(inputCols=["depart_bucket"], outputCols=["depart_dummy"])

# One-hot encode the bucketed departure times
flights = onehot.fit(flights).transform(flights)
flights.select("depart", "depart_bucket", "depart_dummy").show(5)

# features
assembler = VectorAssembler(
    inputCols=[
        "km",
        "org_dummy",
        "depart_dummy",
    ],
    outputCol="features",
)

flights = assembler.transform(flights)

# schema
flights.printSchema()

flights_train, flights_test = flights.randomSplit([0.7, 0.3], seed=42)
# Create a regression object and train on training data - Lasso model (λ = 1, α = 1)
regression = LinearRegression(labelCol="duration", regParam=1, elasticNetParam=1).fit(
    flights_train
)

# Create predictions for the testing data and take a look at the predictions
predictions = regression.transform(flights_test)
predictions.select("duration", "prediction").show(5, False)

# Calculate the RMSE
rmse = RegressionEvaluator(labelCol="duration", predictionCol="prediction").evaluate(
    predictions
)
print(f"test RMSE: {rmse}")

# Number of zero coefficients
zero_coeff = sum([beta == 0 for beta in regression.coefficients])
print("Number of coefficients equal to 0:", zero_coeff)

# Average minutes on ground for all base/ref level of categorical vars - at OGG for flights departing between 21:00 and 24:00
avg_eve_ogg = regression.intercept
print(avg_eve_ogg)

# Average minutes on ground at OGG for flights departing between 03:00 and 06:00
avg_night_ogg = regression.intercept + regression.coefficients[9]
print(avg_night_ogg)

# Average minutes on ground at JFK for flights departing between 03:00 and 06:00
avg_night_jfk = (
    regression.intercept + regression.coefficients[3] + regression.coefficients[9]
)
print(avg_night_jfk)
