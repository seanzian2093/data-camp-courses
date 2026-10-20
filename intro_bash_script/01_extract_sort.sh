#!/bin/bash
# Extract the winners from the soccer_scores.csv file, sort them, and count unique occurrences with `-c`
cat data/soccer_scores.csv | cut -d',' -f2 | tail -n +2 | sort | uniq -c
