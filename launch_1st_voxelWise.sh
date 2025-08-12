#!/bin/bash

# =============================================================================
# First-Level Voxel-Wise fMRI Analysis Job Launcher
# =============================================================================
#
# This script launches all SLURM jobs for first-level voxel-wise fMRI analysis
# across different experimental phases.
#
# Usage:
#   ./launch_1st_voxelWise.sh                    # Launch all jobs
#   bash launch_1st_voxelWise.sh                 # Launch all jobs
#   chmod +x launch_1st_voxelWise.sh && ./launch_1st_voxelWise.sh  # Make executable and run
#
# Prerequisites:
#   1. Run create_1st_voxelWise.py first to generate SLURM scripts
#   2. Ensure SLURM is available on your system
#   3. Make sure you have appropriate permissions to submit jobs
#
# What it does:
#   - Finds all SLURM scripts in phase2/ and phase3/ directories
#   - Submits each script using sbatch
#   - Processes phases sequentially for better organization
#
# Output:
#   - All jobs submitted to SLURM queue
#   - Progress messages showing which phase and scripts are being processed
#
# =============================================================================

# Directory containing SLURM scripts created by create_1st_voxelWise.py
SCRIPTS_BASE_DIR="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/firstLevel"

# Loop through phases and submit all .sh files in each phase directory
for phase in 2 3; do
    phase_dir="$SCRIPTS_BASE_DIR/phase$phase"
    if [ -d "$phase_dir" ]; then
        echo "Processing phase$phase..."
        for script in "$phase_dir"/*.sh; do
            if [ -f "$script" ]; then
                echo "Submitting: $script"
                sbatch "$script"
            fi
        done
    fi
done
