# Import SparkSession from pyspark.sql
from pyspark.sql import SparkSession

# Create my_spark
spark = SparkSession.builder.appName("feature_engineering").getOrCreate()

# Get Spark Context
sc = spark.sparkContext
sc.setLogLevel("ERROR")

if __name__ == "__main__":

    # Print my_spark
    print(spark)

    # Print Spark Context
    print(sc)
