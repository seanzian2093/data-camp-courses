from utils import spark

# Import the pyspark.sql.types library
from pyspark.sql.types import *
import pyspark.sql.functions as F

# Define a new schema using the StructType method
people_schema = StructType(
    [
        # Define a StructField for each field
        StructField("name", StringType(), False),
        StructField("age", IntegerType(), False),
        StructField("city", StringType(), False),
    ]
)

# Load the CSV file
aa_dfw_df = (
    spark.read.format("csv")
    .options(Header=True)
    .load("data/AA_DFW_2017_Departures_Short.csv.gz")
)

# Add the airport column using the F.lower() method - lazily evaluated
aa_dfw_df = aa_dfw_df.withColumn("airport", F.lower(aa_dfw_df["Destination Airport"]))

# Show the DataFrame - only now the transformations are executed
aa_dfw_df.show()

df1 = (
    spark.read.format("csv")
    .options(Header=True)
    .load("data/AA_DFW_2017_Departures_Short.csv.gz")
)
df2 = (
    spark.read.format("csv")
    .options(Header=True)
    .load("data/AA_DFW_2016_Departures_Short.csv.gz")
)
# View the row count of df1 and df2
print("df1 Count: %d" % df1.count())
print("df2 Count: %d" % df2.count())

# Combine the DataFrames into one
df3 = df1.union(df2)

# Save the df3 DataFrame in Parquet format
df3.write.parquet("data/AA_DFW_ALL.parquet", mode="overwrite")

# Read the Parquet file into a new DataFrame and run a count
print(spark.read.parquet("data/AA_DFW_ALL.parquet").count())

# Read the Parquet file into flights_df
flights_df = spark.read.parquet("data/AA_DFW_ALL.parquet")

# Register the temp table
flights_df.createOrReplaceTempView("flights")

# Run a SQL query of the average flight duration
# avg_duration = spark.sql("SELECT avg(flight_duration) from flights").collect()[0]

# Use backticks for column names with spaces or special characters
avg_duration = spark.sql(
    "SELECT avg(`Actual elapsed time (Minutes)`) from flights"
).collect()[0]

print("The average flight time is: %d" % avg_duration)
