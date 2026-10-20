#!/bin/bash

# Extract the winners from the soccer_scores.csv file
# -d',' — field delimiter (comma, since it's a CSV)
# -f1,2 — select fields 1 and 2
# -n +2 means "start at line 2", not "last 2 lines" (which would be -n 2)
cat data/soccer_scores.csv | cut -d',' -f1,2 | tail -n +2 | grep 1959