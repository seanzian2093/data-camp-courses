# Import SparkSession from pyspark.sql
from pyspark.sql import SparkSession

# Create my_spark
spark = SparkSession.builder.appName("clean_data").getOrCreate()

# Get Spark Context
sc = spark.sparkContext
sc.setLogLevel("ERROR")

if __name__ == "__main__":
    # Print my_spark
    print(spark)

    # Print Spark Context
    print(sc)

    # Prepare data - partitioned by file
    aa_dfw_df_full = spark.read.csv(
        "data/AA_DFW_201*_Departures_Short.csv.gz", header=True, inferSchema=True
    )
    print("Total rows: %d" % aa_dfw_df_full.count())

    # Combine and save to a folder, one file for one partition, directly setting file name is not supported
    aa_dfw_df_full.coalesce(1).write.csv(
        "data/AA_DFW_ALL", compression="gzip", mode="overwrite"
    )

    import glob
    import shutil

    # Find the part file
    part_file = glob.glob("data/AA_DFW_ALL/part-*.csv.gz")[0]
    shutil.move(part_file, "data/AA_DFW_ALL.csv.gz")
