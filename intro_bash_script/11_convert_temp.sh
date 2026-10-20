#!/bin/bash

# Get first ARGV into variable
temp_f=$1

# Subtract 32
# `scale=2` ensures the result has 2 decimal places
# `| bc` pipes the expression to the bc calculator which performs the calculation
temp_f2=$(echo "scale=2; $temp_f - 32" | bc)

# Multiply by 5/9
temp_c=$(echo "scale=2; $temp_f2 * 5 / 9" | bc)

# Print the celsius temp
echo $temp_c