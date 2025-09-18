import matplotlib.pyplot as plt
from utils import spark

# Load the Dataframe
file_path = "data/Fifa2018_dataset.csv"
fifa_df = spark.read.csv(file_path, header=True, inferSchema=True)

# Check the schema of columns
fifa_df.printSchema()

# Show the first 10 observations
# fifa_df.show(10)

# Print the total number of rows
print(f"There are {fifa_df.count()} rows in the fifa_df DataFrame")

# Create a temporary view of fifa_df
fifa_df.createOrReplaceTempView("fifa_df_table")

# Construct the "query"
query = '''SELECT Age FROM fifa_df_table WHERE Nationality == "Germany"'''

# Apply the SQL "query"
fifa_df_germany_age = spark.sql(query)

# Generate basic statistics
fifa_df_germany_age.describe().show()

# Convert fifa_df to fifa_df_germany_age_pandas DataFrame
fifa_df_germany_age_pandas = fifa_df_germany_age.toPandas()

# Plot the 'Age' density of Germany Players
fifa_df_germany_age_pandas.plot(kind="density")
plt.show()
