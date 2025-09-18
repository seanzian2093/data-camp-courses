from utils import spark
import ast
import re
from pyspark.ml.regression import GBTRegressor, GBTRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import Vectors, VectorUDT, SparseVector
from pyspark.sql.functions import from_json, udf
from pyspark.sql.types import ArrayType, IntegerType, DoubleType, StructType

# Read the file into a dataframe
df = spark.read.csv("data/train.txt", header=True, inferSchema=True, sep="|")
cols_to_drop = [col for col in df.columns if re.match(r"^_c", col)]
df = df.drop(*cols_to_drop)
print(df.dtypes)

# Convert the string representation of the list into an actual array of integers - from_json does not work
# df = df.withColumn("vector", from_json(df["features"], ArrayType(DoubleType())))
# df.show(5, truncate=False)

# UDF to convert array to Vector - does not work
# parse_list_udf = udf(lambda x: ast.literal_eval(x), ArrayType(DoubleType()))
# df = df.withColumn("vector", parse_list_udf(df["features"]))
# df.show(5, truncate=False)
# array_to_vector_udf = udf(lambda arr: Vectors.dense(arr), VectorUDT())
# train_df = df.withColumn("features", array_to_vector_udf(df["vector"]))


def parse_sparse_vector(s):
    # Extract numbers using regex
    match = re.match(r"\((\d+),\[(.*?)\],\[(.*?)\]\)", s)
    if not match:
        return None
    size = int(match.group(1))
    indices = [int(i) for i in match.group(2).split(",") if i]
    values = [float(v) for v in match.group(3).split(",") if v]
    return SparseVector(size, indices, values)


parse_sparse_vector_udf = udf(parse_sparse_vector, VectorUDT())

# Suppose your DataFrame has a column "features_str" with these strings
train_df = df.withColumn("features", parse_sparse_vector_udf(df["features"]))

print(train_df.dtypes)
# train_df.show(5, truncate=False)

# Train a Gradient Boosted Trees (GBT) model.
gbt = GBTRegressor(
    featuresCol="features",
    labelCol="SALESCLOSEPRICE",
    predictionCol="Prediction_Price",
    seed=42,
)

# Train model.
model = gbt.fit(train_df)
gbt_predictions = model.transform(train_df)


# Select columns to compute test error
evaluator = RegressionEvaluator(
    labelCol="SALESCLOSEPRICE", predictionCol="Prediction_Price"
)
# Dictionary of model predictions to loop over
models = {
    "Gradient Boosted Trees": gbt_predictions,
    # "Random Forest Regression": rfr_predictions,
}
for key, preds in models.items():
    # Create evaluation metrics
    rmse = evaluator.evaluate(preds, {evaluator.metricName: "rmse"})
    r2 = evaluator.evaluate(preds, {evaluator.metricName: "r2"})

    # Print Model Metrics
    print(key + " RMSE: " + str(rmse))
    print(key + " R^2: " + str(r2))

# Save model
model.save("gbt_no_listprice")

# Load model
loaded_model = GBTRegressionModel.load("gbt_no_listprice")
