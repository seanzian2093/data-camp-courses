from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.master("local[*]")
    .appName("sql_spark")
    .config("spark.driver.extraJavaOptions", "-Xss4m")
    .config("spark.executor.extraJavaOptions", "-Xss4m")
    .getOrCreate()
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
