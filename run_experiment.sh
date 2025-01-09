#!/bin/bash

# Record start time
START_TIME=$(date +%s)

# run script
poetry run python ./main.py

# Record end time and compute total duration
END_TIME=$(date +%s)
echo "Total Execution Time: $(($END_TIME - $START_TIME)) seconds"