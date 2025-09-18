from utils import spark, calculate_roem_spark, id_to_index
from pyspark.sql.functions import col, avg, min
from pyspark.ml.recommendation import ALS

# Read data from CSV file
msd = spark.read.csv(
    "data/triplets_file.csv", header=True, inferSchema=True, nullValue="null"
)

msd = msd.withColumnRenamed("listen_count", "num_plays")

msd = id_to_index(msd, user_col="user_id", new_col="userId")
msd = id_to_index(msd, user_col="song_id", new_col="songId")
print("MSD DataFrame:")
msd.show()

# Count the number of distinct userIds
user_count = msd.select("userId").distinct().count()
print("Number of users: ", user_count)

# Count the number of distinct songIds
song_count = msd.select("songId").distinct().count()
print("Number of songs: ", song_count)

# Avg num implicit ratings per songs
print("Average implicit ratings per song: ")
msd.filter(col("num_plays") > 0).groupBy("songId").count().select(avg("count")).show()

# Min num implicit ratings from a user
print("Minimum implicit ratings from a user: ")
msd.filter(col("num_plays") > 0).groupBy("userId").count().select(min("count")).show()

# Avg num implicit ratings for users
print("Average implicit ratings per user: ")
msd.filter(col("num_plays") > 0).groupBy("userId").count().select(avg("count")).show()

# ALS param grid
# ranks = [10, 20, 30, 40]
# maxIters = [10, 20, 30, 40]
# regParams = [0.05, 0.1, 0.15]
# alphas = [20, 40, 60, 80]
ranks = [10]
maxIters = [10]
regParams = [0.05]
alphas = [20]
model_list = []

# For loop will automatically create and store ALS models
for r in ranks:
    for mi in maxIters:
        for rp in regParams:
            for a in alphas:
                model_list.append(
                    ALS(
                        userCol="userId",
                        itemCol="songId",
                        ratingCol="num_plays",
                        rank=r,
                        maxIter=mi,
                        regParam=rp,
                        alpha=a,
                        coldStartStrategy="drop",
                        nonnegative=True,
                        implicitPrefs=True,
                    )
                )

# Print the model list, and the length of model_list
print("Length of model_list: ", len(model_list))

# Split the data into training and test sets
(training, test) = msd.randomSplit([0.8, 0.2])

# Building 5 folds within the training set.
train1, train2, train3, train4, train5 = training.randomSplit(
    [0.2, 0.2, 0.2, 0.2, 0.2], seed=1
)
fold1 = train2.union(train3).union(train4).union(train5)
fold2 = train3.union(train4).union(train5).union(train1)
fold3 = train4.union(train5).union(train1).union(train2)
fold4 = train5.union(train1).union(train2).union(train3)
fold5 = train1.union(train2).union(train3).union(train4)

foldlist = [
    (fold1, train1),
    (fold2, train2),
    (fold3, train3),
    (fold4, train4),
    (fold5, train5),
]

# v_fitted_model = model_list[0].fit(training)
# v_predictions = v_fitted_model.transform(test)
# print(f"ROEM: {calculate_roem_spark(v_predictions)}")
# v_predictions.printSchema()

# Empty list to fill with ROEMs from each model
ROEMS = []

# Loops through all models and all folds
for model in model_list:
    for ft_pair in foldlist:

        # Fits model to fold within training data
        fitted_model = model.fit(ft_pair[0])

        # Generates predictions using fitted_model on respective CV test data
        predictions = fitted_model.transform(ft_pair[1])

        # Generates and prints a ROEM metric CV test data
        r = calculate_roem_spark(predictions)
        print("ROEM: ", r)

    # Fits model to all of training data and generates preds for test data
    v_fitted_model = model.fit(training)
    v_predictions = v_fitted_model.transform(test)
    v_ROEM = calculate_roem_spark(v_predictions)

    # Adds validation ROEM to ROEM list
    ROEMS.append(v_ROEM)
    print("Validation ROEM: ", v_ROEM)
