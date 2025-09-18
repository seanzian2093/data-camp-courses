from utils import spark

# Name of the Spark application instance
app_name = spark.conf.get("spark.app.name")

# Driver TCP port
driver_tcp_port = spark.conf.get("spark.driver.port")

# Number of join partitions
num_partitions = spark.conf.get("spark.sql.shuffle.partitions")

# Show the results
print("Name: %s" % app_name)
print("Driver TCP port: %s" % driver_tcp_port)
print("Number of partitions: %s" % num_partitions)

# Configure Spark to use 500 partitions
spark.conf.set("spark.sql.shuffle.partitions", 500)
print(
    "Afterwards, number of partitions: %s"
    % spark.conf.get("spark.sql.shuffle.partitions")
)
