#!/bin/bash

# Define the list of sgda_learning_rates
learning_rates=(0.0001 0.0003 0.0005 0.0007 0.0009 0.001 0.003 0.005 0.01 0.05)

# Loop over each rate and run the main.py script with that rate
for rate in "${learning_rates[@]}"
do
    python main.py --sgda_learning_rate $rate
done
