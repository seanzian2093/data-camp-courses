import matplotlib.pyplot as plt
from utils import sc, spark

sample_list = [("Mona", 20), ("Jennifer", 34), ("John", 20), ("Jim", 26)]
# Create an RDD from the list
rdd = sc.parallelize(sample_list)

# Create a PySpark DataFrame
names_df = spark.createDataFrame(rdd, schema=["Name", "Age"])

# Check the type of names_df
print("The type of names_df is", type(names_df))

# Create an DataFrame from file_path
file_path = "data/people.csv"
people_df = spark.read.csv(file_path, header=True, inferSchema=True)

# Check the type of people_df
print("The type of people_df is", type(people_df))

# Print the first 10 observations
people_df.show(10)

# Count the number of rows
print("There are {} rows in the people_df DataFrame.".format(people_df.count()))

# Count the number of columns and print their names
print(
    f"There are {len(people_df.columns)} columns in the people_df DataFrame and their names are {people_df.columns}"
)

# Select name, sex and date of birth columns
people_df_sub = people_df.select("name", "sex", "date of birth")

# Print the first 10 observations from people_df_sub
people_df_sub.show(10)

# Remove duplicate entries from people_df_sub
people_df_sub_nodup = people_df_sub.dropDuplicates()

# Count the number of rows
print(
    f"There were {people_df_sub.count()} rows before removing duplicates, and {people_df_sub_nodup.count()} rows after removing duplicates"
)

# Filter people_df to select females
people_df_female = people_df.filter(people_df.sex == "female")

# Filter people_df to select males
people_df_male = people_df.filter(people_df.sex == "male")

# Count the number of rows
print(
    f"There are {people_df_female.count()} rows in the people_df_female DataFrame and {people_df_male.count()} rows in the people_df_male DataFrame"
)

# Create a temporary table "people" - becasue PySpark SQL queries are run against tables, not DataFrames
people_df.createOrReplaceTempView("people")

# Construct a query to select the names of the people from the temporary table "people"
query = """SELECT name FROM people"""

# Assign the result of Spark's query which is always a DataFrame, to people_df_names
people_df_names = spark.sql(query)

# Print the top 10 names of the people
people_df_names.show(10)

# Filter the people table to select female sex
people_female_df = spark.sql('SELECT * FROM people WHERE sex=="female"')

# Filter the people table DataFrame to select male sex
people_male_df = spark.sql('SELECT * from people where sex=="male"')

# Count the number of rows in both people_df_female and people_male_df DataFrames
print(
    f"There are {people_female_df.count()} rows in the people_female_df and {people_male_df.count()} rows in the people_male_df DataFrames"
)

# Check the column names of names_df
print("The column names of names_df are", names_df.columns)

# Convert to Pandas DataFrame
df_pandas = names_df.toPandas()

# Create a horizontal bar plot
df_pandas.plot(kind="barh", x="Name", y="Age", colormap="winter_r")
plt.show()
