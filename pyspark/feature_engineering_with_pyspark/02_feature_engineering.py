from utils import spark
import seaborn as sns
import matplotlib.pyplot as plt
import stats
from pyspark.sql.functions import (
    col,
    to_date,
    dayofweek,
    split,
    year,
    lag,
    datediff,
    when,
    explode,
    coalesce,
    first,
    lit,
)
from pyspark.sql.window import Window
from pyspark.ml.feature import Binarizer, Bucketizer, OneHotEncoder, StringIndexer

# Read the file into a dataframe
df = spark.read.csv(
    "data/2017_StPaul_MN_Real_Estate.csv", header=True, inferSchema=True
)
df = df.withColumnRenamed("No.", "NO")

# Lot size in square feet
acres_to_sqfeet = 43560
df = df.withColumn("LOT_SIZE_SQFT", df["ACRES"] * acres_to_sqfeet)

# Create new column YARD_SIZE
df = df.withColumn("YARD_SIZE", df["LOT_SIZE_SQFT"] - df["FOUNDATIONSIZE"])

# Corr of ACRES vs SALESCLOSEPRICE
print("Corr of ACRES vs SALESCLOSEPRICE: " + str(df.corr("SALESCLOSEPRICE", "ACRES")))
# Corr of FOUNDATIONSIZE vs SALESCLOSEPRICE
print(
    "Corr of FOUNDATIONSIZE vs SALESCLOSEPRICE: "
    + str(df.corr("SALESCLOSEPRICE", "FOUNDATIONSIZE"))
)
# Corr of YARD_SIZE vs SALESCLOSEPRICE
print(
    "Corr of YARD_SIZE vs SALESCLOSEPRICE: "
    + str(df.corr("SALESCLOSEPRICE", "YARD_SIZE"))
)

# ASSESSED_TO_LIST
df = df.withColumn("ASSESSED_TO_LIST", df["ASSESSEDVALUATION"] / df["LISTPRICE"])
df[["ASSESSEDVALUATION", "LISTPRICE", "ASSESSED_TO_LIST"]].show(5)
# TAX_TO_LIST
df = df.withColumn("TAX_TO_LIST", df["TAXES"] / df["LISTPRICE"])
df[["TAX_TO_LIST", "TAXES", "LISTPRICE"]].show(5)
# BED_TO_BATHS
df = df.withColumn("BED_TO_BATHS", df["BEDROOMS"] / df["BATHSTOTAL"])
df[["BED_TO_BATHS", "BEDROOMS", "BATHSTOTAL"]].show(5)

# Create new feature by adding two features together
df = df.withColumn("Total_SQFT", df["SQFTBELOWGROUND"] + df["SQFTABOVEGROUND"])

# Create additional new feature using previously created feature
df = df.withColumn("BATHS_PER_1000SQFT", df["BATHSTOTAL"] / (df["Total_SQFT"] / 1000))
df[["BATHS_PER_1000SQFT"]].describe().show()

# Sample and create pandas dataframe
pandas_df = df.sample(False, 0.5, 0).toPandas()


# Linear model plots
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2


sns.jointplot(
    x="Total_SQFT",
    y="SalesClosePrice",
    data=pandas_df,
    kind="reg",
    # stat_func=r2
)
# plt.show()
sns.jointplot(
    x="BATHS_PER_1000SQFT",
    # case sensitive
    # y="SALESCLOSEPRICE",
    y="SalesClosePrice",
    data=pandas_df,
    kind="reg",
    # stat_func=r2,
)
# plt.show()


# Convert to date type
df = df.withColumn(
    # "LISTDATE", to_date(split(df["LISTDATE"], " ").getItem(0), "M/d/yyyy")
    "LISTDATE",
    to_date(split("LISTDATE", " ").getItem(0), "M/d/yyyy"),
)

# Get the day of the week
df = df.withColumn("List_Day_of_Week", dayofweek(df["LISTDATE"]))

# Sample and convert to pandas dataframe
sample_df = df.sample(False, 0.5, 42).toPandas()

# Plot count plot of of day of week
sns.countplot(x="List_Day_of_Week", data=sample_df)
# plt.show()

price_df = spark.read.csv(
    "data/median_price.txt", header=True, inferSchema=True, sep="|"
)
import re

# Get columns matching pattern
cols_to_drop = [col for col in price_df.columns if re.match(r"^_c", col)]

# Drop those columns
price_df = price_df.drop(*cols_to_drop)
price_df.show(3)

# Create year column
# df = df.withColumn("list_year", year(df["LISTDATE"]))
df = df.withColumn("list_year", year("LISTDATE"))

# Adjust year to match
df = df.withColumn("report_year", (df["list_year"] - 1))

# Create join condition
condition = [df["CITY"] == price_df["City"], df["report_year"] == price_df["Year"]]

# Join the dataframes together
df = df.join(price_df, on=condition, how="left")
# Inspect that new columns are available
df[["MedianHomeValue"]].show()

mort_df = spark.read.csv("data/mort.txt", header=True, inferSchema=True, sep="|")

# Get columns matching pattern
cols_to_drop = [col for col in mort_df.columns if re.match(r"^_c", col)]

# Drop those columns
mort_df = mort_df.drop(*cols_to_drop)
mort_df.show(3)


# Cast data type
mort_df = mort_df.withColumn("DATE", to_date("DATE"))

# Create window
w = Window().orderBy(mort_df["DATE"])
# Create lag column, ie. previous row value
mort_df = mort_df.withColumn("DATE-1", lag("DATE", 1).over(w))

# Calculate difference between date columns
mort_df = mort_df.withColumn("Days_Between_Report", datediff("DATE", "DATE-1"))
# Print results
mort_df.select("Days_Between_Report").distinct().show()

# Create boolean conditions for string matches
has_attached_garage = df["GARAGEDESCRIPTION"].like("%Attached Garage%")
has_detached_garage = df["GARAGEDESCRIPTION"].like("%Detached Garage%")

# Conditional value assignment
df = df.withColumn(
    "has_attached_garage",
    (when(has_attached_garage, 1).when(has_detached_garage, 0).otherwise(None)),
)

# Inspect results
df[["GARAGEDESCRIPTION", "has_attached_garage"]].show(truncate=100)

# Convert string to list-like array
df = df.withColumn("garage_list", split("GARAGEDESCRIPTION", ", "))
df = df.withColumn("constant_val", lit(1))

# Explode the values into new records - one record per item in the list
ex_df = df.withColumn("ex_garage_list", explode("garage_list"))

# Inspect the values
ex_df[["ex_garage_list"]].distinct().show(100, truncate=50)

# Pivot - like a pivot table in Excel
# ex_df.groupBy("NO"): Groups the DataFrame by the column "NO".
# .pivot("ex_garage_list"): Pivots the DataFrame, creating new columns for each unique value in "ex_garage_list".
# .agg(coalesce(first("constant_val"))): For each group and pivoted column, aggregates using the first non-null value of "constant_val" (if any).
# Result:
# This creates a wide DataFrame where each row is a unique "NO", and each column is a unique garage type from "ex_garage_list", filled with the first non-null "constant_val" for that group. If no value exists, it will be null.
piv_df = (
    ex_df.groupBy("NO").pivot("ex_garage_list").agg(coalesce(first("constant_val")))
)

# Join the dataframes together and fill null
joined_df = df.join(piv_df, on="NO", how="left")

# Columns to zero fill
zfill_cols = piv_df.columns

# Zero fill the pivoted values
zfilled_df = joined_df.fillna(0, subset=zfill_cols)


# check dtypes
# print(df.dtypes)
# print(zfilled_df.dtypes)
df = zfilled_df.withColumn("List_Day_of_Week", col("List_Day_of_Week").cast("double"))
# Create the transformer
binarizer = Binarizer(
    threshold=5.0, inputCol="List_Day_of_Week", outputCol="Listed_On_Weekend"
)

# Apply the transformation to df
df = binarizer.transform(df)

# Verify transformation
df[["List_Day_of_Week", "Listed_On_Weekend"]].show()

# Create the bucket splits and bucketizer
splits = [0, 1, 2, 3, 4, 5, float("Inf")]
buck = Bucketizer(splits=splits, inputCol="BEDROOMS", outputCol="bedrooms")

# Apply the transformation to df: df_bucket
df_bucket = buck.transform(df)

# Display results
df_bucket[["BEDROOMS", "bedrooms"]].show()

# Map strings to numbers with string indexer
string_indexer = StringIndexer(
    inputCol="SCHOOLDISTRICTNUMBER", outputCol="School_Index"
)
indexed_df = string_indexer.fit(df).transform(df)

# Onehot encode indexed values
encoder = OneHotEncoder(inputCol="School_Index", outputCol="School_Vec")

# must fit and then transform - for older version, just transform, i.e., 2.3.1
encoder_model = encoder.fit(indexed_df)
encoded_df = encoder_model.transform(indexed_df)

# Inspect the transformation steps
encoded_df[["SCHOOLDISTRICTNUMBER", "School_Index", "School_Vec"]].show(truncate=100)
