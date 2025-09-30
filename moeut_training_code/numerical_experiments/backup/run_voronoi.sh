#!/bin/bash

# Set the PYTHONPATH to include the current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the Voronoi-style error experiment
python3 backup/main_voronoi.py
