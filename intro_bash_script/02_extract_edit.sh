#!/bin/bash

# Replace team names in the soccer_scores.csv file and save the result to soccer_scores_edited.csv
# g for global replacement
cat data/soccer_scores.csv | sed 's/Cherno/Cherno City/g' | sed 's/Arda/Arda United/g' > soccer_scores_edited.csv