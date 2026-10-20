#!/bin/bash

# Extract Accuracy from first ARGV element

# sed is the stream editor, a tool for processing and transforming text
# 's/.* //' is a substitution command that:
# s/ means "substitute"
# .*  matches any characters (zero or more) followed by a space (the regex .*  is greedy, so it matches up to the last space in the line)
# // replaces the matched pattern with nothing (empty string)
accuracy=$(grep Accuracy $1 | sed 's/.* //')

if [ "$accuracy" -ge 90 ]; then
    mv $1 data/good_models/
fi

if [ "$accuracy" -lt 90 ]; then
    mv $1 data/bad_models/
fi
