from utils import spark
from pyspark.sql.functions import split, explode
from pyspark.ml.feature import StringIndexer

# Load the dataframe
# df = spark.read.parquet("data/sherlock.parquet")
df = spark.read.load("data/sherlock.parquet")

# Filter and show the first 5 rows
df.where("id > 70").show(5, truncate=False)

# Split the clause column into a column called words
clauses_df = spark.read.csv(
    "data/sherlock.txt", sep="\n", header=False, inferSchema=True
)
clauses_df = clauses_df.withColumnRenamed("_c0", "clause")
split_df = clauses_df.select(split("clause", " ").alias("words"))
split_df.show(5, truncate=False)

# Explode the words column into a column called word
# exploded_df = split_df.select(explode(col("words")).alias("word"))
exploded_df = split_df.select(explode("words").alias("word"))
exploded_df.show(10)

# Count the resulting number of rows in exploded_df
print("\nNumber of rows: ", exploded_df.count())

text_df = spark.read.csv("data/text_df.txt", sep="|", header=True, inferSchema=True)
text_df.drop("_c0", "_c4").show(5, truncate=False)

# Repartition text_df into 12 partitions on 'chapter' column
text_df = text_df.repartition(12, "chapter")

# Prove that repart_df has 12 partitions
text_df.rdd.getNumPartitions()

# Add part

indexer = StringIndexer(inputCol="chapter", outputCol="part")
text_df = indexer.fit(text_df).transform(text_df)
# Create a temp view
text_df.createOrReplaceTempView("text")
# Find the top 10 sequences of five words
query = """
SELECT w1, w2, w3, w4, w5, COUNT(*) AS count FROM (
   SELECT word AS w1,
   LEAD(word, 1) OVER(partition by part order by id ) AS w2,
   LEAD(word, 2) OVER(partition by part order by id ) AS w3,
   LEAD(word, 3) OVER(partition by part order by id ) AS w4,
   LEAD(word, 4) OVER(partition by part order by id ) AS w5
   FROM text
)
GROUP BY w1, w2, w3, w4, w5
ORDER BY count DESC
LIMIT 10
"""
spark.sql(query).show()

# Unique 5-tuples sorted in descending order
query = """
SELECT distinct w1, w2, w3, w4, w5 FROM (
   SELECT word AS w1,
   LEAD(word,1) OVER(PARTITION BY part ORDER BY id ) AS w2,
   LEAD(word,2) OVER(PARTITION BY part ORDER BY id ) AS w3,
   LEAD(word,3) OVER(PARTITION BY part ORDER BY id ) AS w4,
   LEAD(word,4) OVER(PARTITION BY part ORDER BY id ) AS w5
   FROM text
)
ORDER BY w1 DESC, w2 DESC, w3 DESC, w4 DESC, w5 DESC 
LIMIT 10
"""
spark.sql(query).show()

#   Most frequent 3-tuple per chapter
subquery = """
SELECT chapter, w1, w2, w3, COUNT(*) as count
FROM
(
    SELECT
    chapter,
    word AS w1,
    LEAD(word, 1) OVER(PARTITION BY chapter ORDER BY id ) AS w2,
    LEAD(word, 2) OVER(PARTITION BY chapter ORDER BY id ) AS w3
    FROM text
)
GROUP BY chapter, w1, w2, w3
ORDER BY chapter, count DESC
"""
spark.sql(subquery).show(5)

query = (
    """
SELECT chapter, w1, w2, w3, count FROM
(
  SELECT
  chapter,
  ROW_NUMBER() OVER (PARTITION BY chapter ORDER BY count DESC) AS row,
  w1, w2, w3, count
  FROM ( %s )
)
WHERE row = 1
ORDER BY chapter ASC
"""
    % subquery
)

spark.sql(query).show()
