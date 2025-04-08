#!/bin/bash

scripts_dir="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/firstLevel"

for phaseID in 2 3; do
  PHASE_DIR="$scripts_dir/phase$phaseID"
  echo "Submitting jobs in: $PHASE_DIR"

  for script in "$PHASE_DIR"/*.sh; do
    echo "Submitting $script"
    bash "$script"
  done
done


