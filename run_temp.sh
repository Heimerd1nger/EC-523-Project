#!/bin/bash

# Loop from kd_T=1 to kd_T=10
for kd_T in {1..10}
do
   echo "Running with kd_T=${kd_T}"
   python3 main.py --checkpoints --kd_T ${kd_T} --bs 128 --sgda_learning_rate 0.005 --unlearning_method "SCRUB"
done