#!/bin/bash

# Create a normal array with the mentioned elements using the declare method
declare -a capital_cities

# Add (append) the elements
# since the array remains in current shell session, if we run this script multiple times, 
# the elements will keep getting appended and the array will grow
capital_cities+=("Sydney")
capital_cities+=("Albany")
capital_cities+=("Paris")

# Print out the entire array
echo ${capital_cities[@]}

# Print out the array length
echo ${#capital_cities[@]}

# but this creates a new array from scratch, rather than appending to an existing one
capital_cities1=("New York" "Toronto" "Markham")

# Print out the entire array
echo ${capital_cities1[@]}

# Printout the first element of the array
echo ${capital_cities1[1]}

# Print out the array length
echo ${#capital_cities1[@]}