#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
	echo "Run this script with Bash 4 or newer, not sh or zsh." >&2
	exit 1
fi

if (( BASH_VERSINFO[0] < 4 )); then
# on this mac, bash version is 3.2.57
	echo "This script requires Bash 4 or newer for associative arrays. Current Bash: $BASH_VERSION" >&2
	exit 1
fi

# Declare associative array with key-value pairs on one line
declare -A model_metrics=([model_accuracy]=98 [model_name]="knn" [model_f1]=0.82)

# Print out the entire array
echo ${model_metrics[@]}

# Print out all the keys
echo ${!model_metrics[@]}