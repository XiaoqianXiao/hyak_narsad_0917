#!/bin/bash

scripts_dir="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flow/groupLevel/whole_brain"

#for phaseID in 2 3; do
for phaseID in 2; do
  PHASE_DIR="$scripts_dir/phase$phaseID"
  echo "Submitting jobs in: $PHASE_DIR"

  for script in "$PHASE_DIR"/*_randomise.sh; do
    echo "Submitting $script"
    sbatch "$script"
  done
done


