#!/bin/bash

# Activate the desired conda environment
module load anaconda
source activate bird

# Run the Python script with the provided parameters
python ../mpc/run.py "$1" "$2"
