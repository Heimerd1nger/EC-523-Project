#!/bin/bash

# Loop from kd_T=1 to kd_T=10
for kd_T in {1..10}
do
   echo "Running with kd_T=${kd_T}"
   python main.py --kd_T ${kd_T} --sgda_epochs 10 --msteps 10
done