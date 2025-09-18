from utils import sc

# Create an RDD from a list of words
RDD = sc.parallelize(["Spark", "is", "a", "framework", "for", "Big Data processing"])

# Print out the type of the created object
print("The type of RDD is", type(RDD))

# Print the file_path
file_path = "README.md"
print("The file_path is", file_path)

# Create a fileRDD from file_path
fileRDD = sc.textFile(file_path)

# Check the type of fileRDD
print("The file type of fileRDD is", type(fileRDD))

# Check the number of partitions in fileRDD
print("Number of partitions in fileRDD is", fileRDD.getNumPartitions())

# Create a fileRDD_part from file_path with 5 partitions
fileRDD_part = sc.textFile(file_path, minPartitions=5)

# Check the number of partitions in fileRDD_part
print("Number of partitions in fileRDD_part is", fileRDD_part.getNumPartitions())

# Create map() transformation to cube numbers
numbRDD = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
cubedRDD = numbRDD.map(lambda x: x**3)

# Collect the results
numbers_all = cubedRDD.collect()

# Print the numbers from numbers_all
for numb in numbers_all:
    print(numb)

# Filter the fileRDD to select lines with Spark keyword
fileRDD_filter = fileRDD.filter(lambda line: "Spark" in line)

# How many lines are there in fileRDD?
print("The total number of lines with the keyword Spark is", fileRDD_filter.count())

# Print the first four lines of fileRDD
for line in fileRDD_filter.take(4):
    print(line)

# Create PairRDD Rdd with key value pairs
Rdd = sc.parallelize([(1, 2), (3, 4), (3, 6), (4, 5)])

# Apply reduceByKey() operation on Rdd - lambda works on two values of the same key, i.e., x and y
Rdd_Reduced = Rdd.reduceByKey(lambda x, y: x + y)

# Iterate over the result and print the output
for num in Rdd_Reduced.collect():
    print("Key {} has {} Counts".format(num[0], num[1]))

# Sort the reduced RDD with the key by descending order
Rdd_Reduced_Sort = Rdd_Reduced.sortByKey(ascending=False)

# Iterate over the result and retrieve all the elements of the RDD
for num in Rdd_Reduced_Sort.collect():
    print("Key {} has {} Counts".format(num[0], num[1]))

# Count the unique keys
total = Rdd.countByKey()

# What is the type of total? - dictionary, and should be small enough to fit in memory
print("The type of total is", type(total))

# Iterate over the total and print the output
for k, v in total.items():
    print("key", k, "has", v, "counts")
