# Import SparkSession from pyspark.sql
from pyspark.sql import SparkSession

# Create spark session on local mode, with all available cores
spark = (
    SparkSession.builder.master("local[*]").appName("machine_learning").getOrCreate()
)

# Get Spark Context
sc = spark.sparkContext
sc.setLogLevel("ERROR")

if __name__ == "__main__":

    # Print my_spark
    print(spark.version)

    # Print Spark Context
    print(sc)

    spark.stop()
