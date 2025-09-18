from utils import spark
from pyspark.sql.functions import to_timestamp, to_date, datediff, lit
from datetime import timedelta

# Read the file into a dataframe
df = spark.read.csv(
    "data/2017_StPaul_MN_Real_Estate.csv", header=True, inferSchema=True
)
df = df.withColumnRenamed("offmarketdate", "OFFMKTDATE")
df = df.withColumn("LISTDATE", to_timestamp("LISTDATE", "M/d/yyyy H:mm"))
df = df.withColumn("OFFMKTDATE", to_timestamp("OFFMKTDATE", "M/d/yyyy H:mm"))

df = df.select("OFFMKTDATE", "DAYSONMARKET", "LISTDATE")

df.printSchema()


def train_test_split_date(df, split_col, test_days=45):
    """Calculate the date to split test and training sets"""
    # Find how many days our data spans
    max_date = df.agg({split_col: "max"}).collect()[0][0]
    min_date = df.agg({split_col: "min"}).collect()[0][0]
    # Subtract an integer number of days from the last date in dataset
    split_date = max_date - timedelta(days=test_days)
    return split_date


# Find the date to use in spitting test and train
split_date = train_test_split_date(df, "OFFMKTDATE")

# Create Sequential Test and Training Sets
train_df = df.where(df["OFFMKTDATE"] < split_date)
test_df = df.where(df["OFFMKTDATE"] >= split_date).where(df["LISTDATE"] <= split_date)

split_date = to_date(lit("2017-12-10"))
# Create Sequential Test set
# test_df = df.where(df['OFFMKTDATE'] >= split_date).where(df['LISTDATE'] <= split_date)

# Create a copy of DAYSONMARKET to review later
test_df = test_df.withColumn("DAYSONMARKET_Original", test_df["DAYSONMARKET"])

# Recalculate DAYSONMARKET from what we know on our split date
test_df = test_df.withColumn("DAYSONMARKET", datediff(split_date, "LISTDATE"))

# Review the difference
test_df[["LISTDATE", "OFFMKTDATE", "DAYSONMARKET_Original", "DAYSONMARKET"]].show()
