#!/bin/bash
# Create three variables from the temp data files' contents
temp_a=$(cat data/region_A)
temp_b=$(cat data/region_B)
temp_c=$(cat data/region_C)

# Print out the three variables
echo "The three temperatures were $temp_a, $temp_b, and $temp_c"
