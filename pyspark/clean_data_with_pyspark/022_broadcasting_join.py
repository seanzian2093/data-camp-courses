from utils import spark
from pyspark.sql.functions import broadcast
import time

# Prepare data - partitioned by file
flights_df = spark.read.csv(
    "data/AA_DFW_201*_Departures_Short.csv.gz", header=True, inferSchema=True
)
print("Total rows in flights_df: %d" % flights_df.count())

airports_df = spark.read.csv("data/airports.csv", header=True, inferSchema=True)
print("Total rows in airports_df: %d" % airports_df.count())

# Join-without broadcasting the flights_df and aiports_df DataFrames
normal_df = flights_df.join(
    airports_df, flights_df["Destination Airport"] == airports_df["code"]
)

# Show the query plan
normal_df.explain()

# Import the broadcast method from pyspark.sql.functions

# Join the flights_df and airports_df DataFrames using broadcasting
broadcast_df = flights_df.join(
    broadcast(airports_df), flights_df["Destination Airport"] == airports_df["code"]
)

# Show the query plan and compare against the original
broadcast_df.explain()

start_time = time.time()
# Count the number of rows in the normal DataFrame
normal_count = normal_df.count()
normal_duration = time.time() - start_time

start_time = time.time()
# Count the number of rows in the broadcast DataFrame
broadcast_count = broadcast_df.count()
broadcast_duration = time.time() - start_time

# Print the counts and the duration of the tests
print("Normal count:\t\t%d\tduration: %f" % (normal_count, normal_duration))
print("Broadcast count:\t%d\tduration: %f" % (broadcast_count, broadcast_duration))
