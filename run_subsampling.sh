#!/bin/bash

# Loop from kd_T=10 to kd_T=100 with an interval of 10
for sub_sample in $(seq 200 20 300)
do
   echo "Running with kd_T=${sub_sample}"
   python main.py --sub_sample ${sub_sample} --sgda_epochs 10 --msteps 10
done
