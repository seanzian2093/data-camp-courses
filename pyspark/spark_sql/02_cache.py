from utils import spark
import time
from pyspark import StorageLevel

# cache() is essentially persist() with the default storage level
# Use persist() when you need specific storage behavior (memory-only, disk-only, etc.)
# Both require an action (like count()) to actually materialize the caching

# Common storage levels:
StorageLevel.MEMORY_ONLY  # Keep in memory only
StorageLevel.MEMORY_AND_DISK  # Memory first, disk if needed
StorageLevel.MEMORY_ONLY_2  # Serialized in memory only
StorageLevel.MEMORY_AND_DISK_2  # Serialized, memory + disk
StorageLevel.DISK_ONLY  # Store only on disk

df1 = spark.read.load("data/sherlock.parquet")
df2 = spark.read.load("data/sherlock2.parquet")
# df2 = spark.read.csv("data/sherlock.txt", sep="\n", header=False, inferSchema=True)


# Utility functions
def prep(df1, df2):
    global begin
    df1.unpersist()
    df2.unpersist()
    begin = time.time()


def print_elapsed():
    print("Overall elapsed : %.1f" % (time.time() - begin))


def run(df, name, elapsed=False):
    start = time.time()
    df.count()
    print("%s : %.1fs" % (name, (time.time() - start)))
    if elapsed:
        print_elapsed()


# Unpersists df1 and df2 and initializes a timer
prep(df1, df2)

# Cache df1
df1.cache()

# Run actions on both dataframes
run(df1, "df1_1st")
run(df1, "df1_2nd")
run(df2, "df2_1st")
run(df2, "df2_2nd", elapsed=True)

# Prove df1 is cached
print(f"df1 is cached: {df1.is_cached}")

# Unpersist df1 and df2 and initializes a timer
prep(df1, df2)

# Persist df2 using memory and disk storage level
df2.persist()

# Run actions both dataframes
run(df1, "df1_1st")
run(df1, "df1_2nd")
run(df2, "df2_1st")
run(df2, "df2_2nd", elapsed=True)

# Unpersist df1 and df2 and initializes a timer
prep(df1, df2)

# Persist df2 using memory and disk storage level
df2.persist(storageLevel=StorageLevel.MEMORY_AND_DISK)

# Run actions both dataframes
run(df1, "df1_1st")
run(df1, "df1_2nd")
run(df2, "df2_1st")
run(df2, "df2_2nd", elapsed=True)

# List the tables
df1.createOrReplaceTempView("table1")
print("Tables:\n", spark.catalog.listTables())

# Cache table1 and Confirm that it is cached
spark.catalog.cacheTable("table1")
print("table1 is cached: ", spark.catalog.isCached("table1"))

# Uncache table1 and confirm that it is uncached
spark.catalog.uncacheTable("table1")
print("table1 is cached: ", spark.catalog.isCached("table1"))
