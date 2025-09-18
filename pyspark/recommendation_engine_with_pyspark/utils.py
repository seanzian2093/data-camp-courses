# Import SparkSession from pyspark.sql
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    array,
    col,
    explode,
    lit,
    struct,
    percent_rank,
)
from pyspark.sql import Window
from pyspark.ml.feature import StringIndexer

# Create spark session on local mode, with all available cores
spark = (
    SparkSession.builder.master("local[*]")
    .appName("recommendation_engine")
    .config("spark.driver.extraJavaOptions", "-Xss4m")
    .config("spark.executor.extraJavaOptions", "-Xss4m")
    .getOrCreate()
)

# Get Spark Context
sc = spark.sparkContext
sc.setLogLevel("ERROR")


def to_long(df, by=["User"]):
    """
    Converts traditional or "wide" dataframe into a "row-based" dataframe, also known as a "dense" or "long" dataframe.

    Parameters:
      - df: array of columns with column names
      - by: name of column which serves as

    Returns: Row-based dataframe with no null values
    """
    cols = [c for c in df.columns if c not in by]
    # Create and explode an array of (column_name, column_value) structs
    kvs = explode(
        array([struct(lit(c).alias("Movie"), col(c).alias("Rating")) for c in cols])
    ).alias("kvs")
    return (
        df.select(by + [kvs])
        .select(by + ["kvs.Movie", "kvs.Rating"])
        .filter("rating IS NOT NULL")
    )


def id_to_index(df, user_col="id", new_col="idx"):
    """
    Replace each unique user_id with a unique integer.
    Args:
        df: PySpark DataFrame with a user_id column
        user_col: name of the user ID column
        new_col: name for the new integer column
    Returns:
        DataFrame with new integer user index column
    """
    indexer = StringIndexer(inputCol=user_col, outputCol=new_col)
    model = indexer.fit(df)
    df_indexed = model.transform(df)
    return df_indexed


def calculate_roem_spark(df):
    df = add_percent_rank(df)
    df = df.withColumn("np*rank", col("num_plays") * col("percRank"))
    numerator = df.groupBy().sum("np*rank").collect()[0][0]
    denominator = df.groupBy().sum("num_plays").collect()[0][0]
    roem = numerator / denominator
    return roem


def add_percent_rank(
    df, order_col="prediction", partition_col=None, new_col="percRank"
):
    """
    Adds a percent rank column to the DataFrame.
    Args:
        df: PySpark DataFrame
        order_col: column to order by for ranking
        partition_col: optional column to partition by (e.g., user)
        new_col: name for the new percent rank column
    Returns:
        DataFrame with percent rank column
    """
    if partition_col:
        window = Window.partitionBy(partition_col).orderBy(order_col)
    else:
        window = Window.orderBy(order_col)
    df_ranked = df.withColumn(new_col, percent_rank().over(window))
    return df_ranked


if __name__ == "__main__":

    # Print my_spark
    print(spark.version)

    # Print Spark Context
    print(sc)

    spark.stop()
