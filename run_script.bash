#!/bin/bash

# Loop 1000 times
count=1500
for i in $(seq $count);
do
   echo "Run #$i"
   python run_sim.py
done
