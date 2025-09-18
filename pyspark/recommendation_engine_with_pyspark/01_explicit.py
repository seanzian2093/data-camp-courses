from utils import spark
from pyspark.sql.functions import col, avg, min
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# Read data from txt file
# ratings = spark.read.csv(
#     "data/ratings.txt", sep="|", header=True, inferSchema=True, nullValue="null"
# )
# ratings = ratings.drop("_c0", "_c5")

# Read data from CSV file
ratings = spark.read.csv(
    "data/ml-latest-small/ratings.csv", header=True, inferSchema=True, nullValue="null"
)

print("ratings DataFrame:")
ratings.show()

# Count the total number of ratings in the dataset
numerator = ratings.select("rating").count()

# Count the number of distinct userIds and distinct movieIds
num_users = ratings.select("userId").distinct().count()
num_movies = ratings.select("movieId").distinct().count()

# Set the denominator equal to the number of users multiplied by the number of movies
denominator = num_users * num_movies

# Divide the numerator by the denominator
sparsity = (1.0 - (numerator * 1.0) / denominator) * 100
print("The ratings dataframe is ", "%.2f" % sparsity + "% empty.")

# View the ratings dataset
ratings.show()

# Filter to show only userIds less than 100
ratings.filter(col("userId") < 100).show()

# Group data by userId, count ratings
ratings.groupBy("userId").count().show()

# Min num ratings for movies
print("Movie with the fewest ratings: ")
ratings.groupBy("movieId").count().select(min("count")).show()

# Avg num ratings per movie
print("Avg num ratings per movie: ")
ratings.groupBy("movieId").count().select(avg("count")).show()

# Min num ratings for user
print("User with the fewest ratings: ")
ratings.groupBy("userId").count().select(min("count")).show()

# Avg num ratings per users
print("Avg num ratings per user: ")
ratings.groupBy("userId").count().select(avg("count")).show()

# Use .printSchema() to see the datatypes of the ratings dataset
ratings.printSchema()

# Tell Spark to convert the columns to the proper data types
ratings = ratings.select(
    ratings.userId.cast("integer"),
    ratings.movieId.cast("integer"),
    ratings.rating.cast("double"),
)

# Call .printSchema() again to confirm the columns are now in the correct format
ratings.printSchema()

# Create test and train set
(train, test) = ratings.randomSplit([0.8, 0.2], seed=1234)

# Create ALS model
als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    nonnegative=True,
    implicitPrefs=False,
)

# Confirm that a model called "als" was created
print(type(als))

# Add hyperparameters and their respective values to param_grid
param_grid = (
    ParamGridBuilder()
    # .addGrid(als.rank, [10, 50, 100, 150])
    .addGrid(als.rank, [10])
    .addGrid(als.maxIter, [50])
    .addGrid(als.regParam, [0.01])
    .build()
)

# Define evaluator as RMSE and print length of evaluator
evaluator = RegressionEvaluator(
    metricName="rmse", labelCol="rating", predictionCol="prediction"
)
print("Num models to be tested: ", len(param_grid))

# Build cross validation using CrossValidator
cv = CrossValidator(
    estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5
)

# Confirm cv was built
print(cv)

# Fit cross validator to the 'train' dataset
model = cv.fit(train)

# Extract best model from the cv model above
best_model = model.bestModel

# Print best_model
print(type(best_model))

# Complete the code below to extract the ALS model parameters
print("**Best Model**")

# Print "Rank"
print("  Rank:", best_model.rank)

# Print "MaxIter"
# print("  MaxIter:", best_model.MaxIter)

# Print "RegParam"
# print("  RegParam:", best_model.RegParam)

test_predictions = best_model.transform(test)
# Calculate and print the RMSE of test_predictions
RMSE = evaluator.evaluate(test_predictions)
print(f"best_model's RMSE: {RMSE}")
