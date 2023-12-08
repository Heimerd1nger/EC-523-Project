#!/bin/bash

# Loop from kd_T=10 to kd_T=100 with an interval of 10
for sub_sample in $(seq 0.4 0.1 1.0)
do
   echo "Running with sample portion=${sub_sample}"
   python3 main.py --checkpoints --bs 256 --sub_sample ${sub_sample} --sgda_learning_rate 0.006 --unlearning_method "SCRUB"
done
