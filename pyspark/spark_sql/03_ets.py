from utils import spark
from pyspark.sql.types import BooleanType, StringType, ArrayType
from pyspark.sql.functions import split, regexp_replace, col, udf, array_contains
from pyspark.ml.feature import CountVectorizer

# Read the pipe-delimited file and convert to arrays
df = (
    spark.read.option("header", "true")
    .option("delimiter", "|")
    .csv("data/ets_df2.txt")
    .select(
        split(regexp_replace(col("doc"), r"[\[\]]", ""), ",").alias("doc"),
        split(regexp_replace(col("in"), r"[\[\]]", ""), ",").alias("in"),
        split(regexp_replace(col("out"), r"[\[\]]", ""), ",").alias("out"),
    )
)

df.show(5, truncate=False)
df.printSchema()

# Returns true if the value is a nonempty vector
nonempty_udf = udf(
    lambda x: True if (x and hasattr(x, "toArray") and x.numNonzeros()) else False,
    BooleanType(),
)

# Returns first element of the array as string
s_udf = udf(
    lambda x: str(x[0]) if (x and type(x) is list and len(x) > 0) else "",
    StringType(),
)

TRIVIAL_TOKENS = {
    "",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "b",
    "c",
    "e",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "pp",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
}

# Show the rows where doc contains the item '5'
df_before = df
# df_before.where(array_contains("doc", "5")).show()

# UDF removes items in TRIVIAL_TOKENS from array
rm_trivial_udf = udf(
    lambda x: list(set(x) - TRIVIAL_TOKENS) if x else x, ArrayType(StringType())
)

# Remove trivial tokens from 'in' and 'out' columns of df2
df_after = df_before.withColumn("in", rm_trivial_udf("in")).withColumn(
    "out", rm_trivial_udf("out")
)

# Show the rows of df_after where doc contains the item '5'
# df_after.where(array_contains("doc", "5")).show()

df = df_after
model = CountVectorizer(inputCol="words", outputCol="vec")
model = model.fit(df.withColumnRenamed("in", "words"))

# Transform df using model
result = (
    model.transform(df.withColumnRenamed("in", "words"))
    .withColumnRenamed("words", "in")
    .withColumnRenamed("vec", "invec")
)
result.drop("doc").show(3, False)

# Add a column based on the out column called outvec
result = (
    model.transform(result.withColumnRenamed("out", "words"))
    .withColumnRenamed("words", "out")
    .withColumnRenamed("vec", "outvec")
)
result.select("invec", "outvec").show(3, False)
