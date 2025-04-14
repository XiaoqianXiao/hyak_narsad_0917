#!/bin/bash

scripts_dir="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flow/groupLevel/whole_brain"

#for phaseID in 2 3; do
for phaseID in 2; do
  for script in "scripts_dir"/*_phase$phaseID*randomise.sh; do
    echo "Submitting $script"
    sbatch "$script"
  done
done


