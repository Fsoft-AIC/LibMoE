#!/bin/bash

# Set the PYTHONPATH to include the current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the Voronoi-style error experiment
python3 numerical_exp/main_voronoi.py
