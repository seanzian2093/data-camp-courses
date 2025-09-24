from utils import spark

schedule = spark.read.csv("data/trainsched.txt", header=True, inferSchema=True)

schedule.createOrReplaceTempView("schedule")

# Inspect the columns in the table schedule
spark.sql("DESCRIBE schedule").show()

# Convert time strings to proper time format- breaking changes in time format parsing after spark 3.0
time_conversion_query = """
SELECT 
    train_id, 
    station, 
    time,
    regexp_replace(regexp_replace(time, 'a$', 'AM'), 'p$', 'PM') AS time_formatted,
    to_timestamp(
        regexp_replace(regexp_replace(time, 'a$', 'AM'), 'p$', 'PM'), 
        'h:mma'
    ) AS time_converted
FROM schedule
ORDER BY train_id, time_converted
"""

schedule = spark.sql(time_conversion_query)
# schedule.printSchema()

# Update the temp view
schedule.createOrReplaceTempView("schedule")

# time difference between current and previous row
prev_time_query = """
SELECT train_id, station, time, time_converted,
lead(time_converted, 1) OVER (PARTITION BY train_id ORDER BY time) AS prev_time,
unix_timestamp(lead(time_converted, 1) OVER (PARTITION BY train_id ORDER BY time)) - unix_timestamp(time_converted) AS diff_min
FROM schedule
"""
schedule = spark.sql(prev_time_query)
schedule.printSchema()
schedule.createOrReplaceTempView("schedule")
schedule.show()

# Add col running_total that sums diff_min col in each group
query1 = """
SELECT train_id, station, time, diff_min,
sum(diff_min) OVER (PARTITION BY train_id ORDER BY time) AS running_total
FROM schedule
"""

spark.sql(query1).show(truncate=False)

# Give the identical result in each command
df = schedule
spark.sql("SELECT train_id, MIN(time) AS start FROM schedule GROUP BY train_id").show()
df.groupBy("train_id").agg({"time": "min"}).withColumnRenamed(
    "min(time)", "start"
).show()

# Print the second column of the result
spark.sql(
    "SELECT train_id, MIN(time), MAX(time) FROM schedule GROUP BY train_id"
).show()
result = df.groupBy("train_id").agg({"time": "min", "time": "max"})
result.show()
print(result.columns[1])
