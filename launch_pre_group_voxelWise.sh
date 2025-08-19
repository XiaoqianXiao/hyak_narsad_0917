#!/bin/bash
# Launch script for pre-group voxel-wise analysis
# This script submits all SLURM scripts in the scripts directory
#
# Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)

# Scripts directory
SCRIPTS_DIR="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/groupLevel/pregroup"

echo "=== Pre-group Voxel-wise Analysis Launcher ==="
echo "Scripts directory: $SCRIPTS_DIR"
echo ""

# Change to scripts directory and submit all .sh files
cd "$SCRIPTS_DIR"
sbatch *.sh

echo ""
echo "✅ All jobs submitted!"
echo "Check status with: squeue -u \$USER"
