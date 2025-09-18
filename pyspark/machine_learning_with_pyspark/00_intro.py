from utils import spark
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# Read data from CSV file
flights = spark.read.csv(
    "data/flights.csv", sep=",", header=True, inferSchema=True, nullValue="NA"
)

# Get number of records
print("The data contain %d records." % flights.count())

# View the first five records
flights.show(5)

# Check column data types
print(flights.dtypes)


# Specify column names and types
schema = StructType(
    [
        StructField("id", IntegerType()),
        StructField("text", StringType()),
        StructField("label", IntegerType()),
    ]
)

# Load data from a delimited file
sms = spark.read.csv("data/sms.csv", sep=";", header=False, schema=schema)

# Print schema of DataFrame
sms.printSchema()
