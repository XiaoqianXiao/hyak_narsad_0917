#!/bin/bash
# Launch script for pre-group voxel-wise analysis
# This script submits all SLURM scripts in the specified scripts directory
#
# Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)
#
# USAGE:
#   # Submit all scripts in default directory
#   bash launch_pre_group_voxelWise.sh
#
#   # Submit scripts in custom directory
#   bash launch_pre_group_voxelWise.sh /path/to/scripts
#
#   # Show help
#   bash launch_pre_group_voxelWise.sh --help

set -e

# Default scripts directory
DEFAULT_SCRIPTS_DIR="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/groupLevel/pregroup"

# Parse arguments
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: $0 [SCRIPTS_DIR]"
    echo ""
    echo "Arguments:"
    echo "  SCRIPTS_DIR    Directory containing SLURM scripts (default: $DEFAULT_SCRIPTS_DIR)"
    echo ""
    echo "Examples:"
    echo "  # Submit all scripts in default directory"
    echo "  $0"
    echo ""
    echo "  # Submit scripts in custom directory"
    echo "  $0 /path/to/scripts"
    echo ""
    echo "  # Show this help"
    echo "  $0 --help"
    exit 0
fi

# Set scripts directory
if [[ -n "$1" ]]; then
    SCRIPTS_DIR="$1"
else
    SCRIPTS_DIR="$DEFAULT_SCRIPTS_DIR"
fi

echo "=== Pre-group Voxel-wise Analysis Launcher ==="
echo "Scripts directory: $SCRIPTS_DIR"
echo ""

# Check if scripts directory exists
if [[ ! -d "$SCRIPTS_DIR" ]]; then
    echo "❌ Scripts directory not found: $SCRIPTS_DIR"
    echo "Please run create_pre_group_voxelWise.py first to generate the scripts."
    exit 1
fi

# Find all .sh files
sh_files=($SCRIPTS_DIR/*.sh)

if [[ ${#sh_files[@]} -eq 0 ]]; then
    echo "❌ No .sh files found in: $SCRIPTS_DIR"
    exit 1
fi

echo "Found ${#sh_files[@]} SLURM scripts:"
for script in "${sh_files[@]}"; do
    echo "  $(basename "$script")"
done
echo ""

# Submit all jobs
echo "🚀 Submitting all jobs..."
cd "$SCRIPTS_DIR"

for script in "${sh_files[@]}"; do
    if [[ -f "$script" ]]; then
        echo "Submitting $(basename "$script")"
        sbatch "$script"
        sleep 1  # Small delay between submissions
    fi
done

echo ""
echo "✅ All jobs submitted successfully!"
echo ""
echo "📊 To monitor jobs:"
echo "  squeue -u \$USER --name='pre_group_*'"
echo ""
echo "🔍 Or check specific job status:"
echo "  squeue -u \$USER --job <job_id>"
