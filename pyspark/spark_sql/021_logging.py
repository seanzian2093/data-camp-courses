import logging
import sys
from utils import spark

# Configure logging to output to standard out with level DEBUG
logging.basicConfig(
    stream=sys.stdout, level=logging.DEBUG, format="%(levelname)s - %(message)s"
)

text_df = spark.read.csv("data/text_df.txt", sep="|", header=True, inferSchema=True)
text_df.drop("_c0", "_c4").show(5, truncate=False)
text_df.createOrReplaceTempView("table1")

# Log columns of text_df as debug message
logging.debug("text_df columns: %s", text_df.columns)

# Log whether table1 is cached as info message
logging.info("table1 is cached: %s", spark.catalog.isCached(tableName="table1"))

# Log first row of text_df as warning message
logging.warning("The first row of text_df:\n %s", text_df.first())

# Log selected columns of text_df as error message
logging.error("Selected columns: %s", text_df.select("id", "word"))

# Uncomment the 5 statements that do NOT trigger text_df
logging.debug("text_df columns: %s", text_df.columns)
logging.info("table1 is cached: %s", spark.catalog.isCached(tableName="table1"))
# logging.warning("The first row of text_df: %s", text_df.first())
logging.error("Selected columns: %s", text_df.select("id", "word"))

# sql query is lazy
logging.info("Tables: %s", spark.sql("show tables").collect())
logging.debug("First row: %s", spark.sql("SELECT * FROM table1 limit 1"))
# logging.debug("Count: %s", spark.sql("SELECT COUNT(*) AS count FROM table1").collect())
