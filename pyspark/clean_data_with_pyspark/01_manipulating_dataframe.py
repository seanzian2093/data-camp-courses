from utils import spark
import pyspark.sql.functions as F
from pyspark.sql.types import StringType

# Load the CSV file
voter_df = (
    spark.read.format("csv")
    .options(Header=True)
    .load("data/DallasCouncilVoters.csv.gz")
)
# Show the distinct VOTER_NAME entries
voter_df.select("VOTER_NAME").distinct().show(40, truncate=False)

# Filter voter_df where the VOTER_NAME is 1-20 characters in length
voter_df = voter_df.filter("length(VOTER_NAME) > 0 and length(VOTER_NAME) < 20")

# Filter out voter_df where the VOTER_NAME contains an underscore
voter_df = voter_df.filter(~F.col("VOTER_NAME").contains("_"))

# Show the distinct VOTER_NAME entries again
voter_df.select("VOTER_NAME").distinct().show(40, truncate=False)

# Add a new column called splits separated on whitespace
voter_df = voter_df.withColumn("splits", F.split(voter_df.VOTER_NAME, "\s+"))

# Create a new column called first_name based on the first item in splits
voter_df = voter_df.withColumn("first_name", voter_df.splits.getItem(0))

# Get the last entry of the splits list and create a column called last_name
voter_df = voter_df.withColumn(
    "last_name", voter_df.splits.getItem(F.size("splits") - 1)
)

# Drop the splits column
# voter_df = voter_df.drop("splits")

# Show the voter_df DataFrame
voter_df.show()

# Add a column to voter_df for any voter with the title **Councilmember**
voter_df = voter_df.withColumn(
    "random_val", F.when(voter_df.TITLE == "Councilmember", F.rand())
)

# Show some of the DataFrame rows, noting whether the when clause worked
voter_df.show()

# Add a column to voter_df for a voter based on their position - automatically overwrites the previous random_val column
voter_df = voter_df.withColumn(
    "random_val",
    F.when(voter_df.TITLE == "Councilmember", F.rand())
    .when(voter_df.TITLE == "Mayor", 2)
    .otherwise(0),
)
# Show some of the DataFrame rows
voter_df.show()

# Use the .filter() clause with random_val
voter_df.filter(voter_df.random_val == 0).show()


def getFirstAndMiddle(names):
    # Return a space separated string of names
    return " ".join(names[:-1])


# Define the method as a UDF
udfFirstAndMiddle = F.udf(getFirstAndMiddle, StringType())

# Create a new column using your UDF
voter_df = voter_df.withColumn(
    "first_and_middle_name", udfFirstAndMiddle(voter_df.splits)
)

# Show the DataFrame
voter_df.show()

# Select all the unique council voters
df = (
    spark.read.format("csv").options(Header=True).load("data/DallasCouncilVotes.csv.gz")
)
voter_df = df.select(df["VOTER NAME"]).distinct()

# Count the rows in voter_df
print("\nThere are %d rows in the voter_df DataFrame.\n" % voter_df.count())

# Add a ROW_ID
voter_df = voter_df.withColumn("ROW_ID", F.monotonically_increasing_id())

# Show the rows with 10 highest IDs in the set
voter_df.orderBy(voter_df.ROW_ID.desc()).show(10)

# Repartition voter_df_single into 1 partitions
voter_df = voter_df.repartition(2)
voter_df_single = voter_df.repartition(1)

# Print the number of partitions in each DataFrame
print(
    "\nThere are %d partitions in the voter_df DataFrame.\n"
    % voter_df.rdd.getNumPartitions()
)
print(
    "\nThere are %d partitions in the voter_df_single DataFrame.\n"
    % voter_df_single.rdd.getNumPartitions()
)

# Add a ROW_ID field to each DataFrame
voter_df = voter_df.withColumn("ROW_ID", F.monotonically_increasing_id())
voter_df_single = voter_df_single.withColumn("ROW_ID", F.monotonically_increasing_id())

# Show the top 10 IDs in each DataFrame
voter_df.orderBy(voter_df.ROW_ID.desc()).show(10)
voter_df_single.orderBy(voter_df_single.ROW_ID.desc()).show(10)
