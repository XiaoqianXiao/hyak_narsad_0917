#!/bin/bash
# Launch script for pre-group voxel-wise analysis
# This script submits all SLURM scripts in the scripts directory
#
# Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)

# Scripts directory
SCRIPTS_DIR="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/groupLevel/pregroup"


# Change to scripts directory and submit all .sh files
cd "$SCRIPTS_DIR"
for i in pre_group_*.sh; do
    sbatch ${i}
done
