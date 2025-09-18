import time
from utils import spark

# Read in the CSV
competitor_df = spark.read.csv(
    "data/190309_DPLC_Competitors_only.csv", header=True, inferSchema=True
)

start_time = time.time()

# Add caching to the unique rows in competitor_df - not cached yet at this point
competitor_df = competitor_df.distinct().cache()

# Count the unique rows in competitor_df, noting how long the operation takes
print(
    "Counting %d rows took %f seconds"
    % (competitor_df.count(), time.time() - start_time)
)

# Count the rows again, noting the variance in time of a cached DataFrame
start_time = time.time()
print(
    "Counting %d rows again took %f seconds"
    % (competitor_df.count(), time.time() - start_time)
)

# Determine if competitor_df is in the cache
print("Is competitor_df cached?: %s" % competitor_df.is_cached)
print("Removing competitor_df from cache")

# Remove competitor_df from the cache
competitor_df.unpersist()

# Check the cache status again
print("Is competitor_df cached?: %s" % competitor_df.is_cached)

# Import the full and split files into DataFrames
full_df = spark.read.csv("data/AA_DFW_ALL.csv.gz")
split_df = spark.read.csv("data/AA_DFW_201*_Departures_Short.csv.gz")

# Print the count and run time for each DataFrame
start_time_a = time.time()
print("Total rows in full DataFrame:\t%d" % full_df.count())
print("Time to run: %f" % (time.time() - start_time_a))

# Splitting input file into multiple parts of equal size boost performance
start_time_b = time.time()
print("Total rows in split DataFrame:\t%d" % split_df.count())
print("Time to run: %f" % (time.time() - start_time_b))
