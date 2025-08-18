#!/bin/bash
# Launch script for pre-group voxel-wise analysis
# This script generates SLURM scripts and launches them
#
# Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)
#
# USAGE:
#   # Generate and launch SLURM scripts for all subjects and phases
#   bash launch_pre_group_voxelWise.sh
#
#   # Process specific subjects
#   bash launch_pre_group_voxelWise.sh --subjects sub-001,sub-002
#
#   # Process specific phases
#   bash launch_pre_group_voxelWise.sh --phases phase2
#
#   # Process specific subjects for specific phases
#   bash launch_pre_group_voxelWise.sh --subjects sub-001,sub-002 --phases phase2,phase3
#
#   # Custom time and memory limits
#   bash launch_pre_group_voxelWise.sh --time 08:00:00 --mem 64G
#
#   # Custom script directory
#   bash launch_pre_group_voxelWise.sh --script-dir /custom/script/path
#
#   # Dry run to see what would be created
#   bash launch_pre_group_voxelWise.sh --dry-run
#
#   # Show help
#   bash launch_pre_group_voxelWise.sh --help

set -e

# Default parameters
OUTPUT_DIR="/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/groupLevel"
SCRIPT_DIR=""
DERIVATIVES_DIR="/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis"
DATA_SOURCE="all"
WORKDIR="/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --script-dir)
            SCRIPT_DIR="$2"
            shift 2
            ;;
        --derivatives-dir)
            DERIVATIVES_DIR="$2"
            shift 2
            ;;
        --workdir)
            WORKDIR="$2"
            shift 2
            ;;
        --subjects)
            SUBJECTS="$2"
            shift 2
            ;;
        --phases)
            PHASES="$2"
            shift 2
            ;;
        --data-source)
            DATA_SOURCE="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --mem)
            MEM="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --output-dir DIR      Output directory for results (default: $OUTPUT_DIR)"
            echo "  --script-dir DIR      Directory for SLURM scripts (default: $SCRIPT_DIR)"
            echo "  --derivatives-dir DIR Derivatives directory (default: $DERIVATIVES_DIR)"
            echo "  --workdir DIR         Work directory for scripts (default: $WORKDIR)"
            echo "  --subjects LIST       Comma-separated list of subjects (e.g., sub-001,sub-002)"
            echo "  --phases LIST         Comma-separated list of phases (e.g., phase2,phase3)"
            echo "  --data-source TYPE    Data source: all, placebo, or guess (default: all)"
            echo "  --time TIME           SLURM time limit (e.g., 08:00:00)"
            echo "  --mem MEM             SLURM memory limit (e.g., 64G)"
            echo "  --dry-run             Show what would be done without executing"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Generate scripts for all subjects and phases"
            echo "  $0"
            echo ""
            echo "  # Generate scripts for specific subjects"
            echo "  $0 --subjects sub-001,sub-002"
            echo ""
            echo "  # Generate scripts for specific data source"
            echo "  $0 --data-source placebo"
            echo ""
            echo "  # Generate scripts with custom parameters"
            echo "  $0 --time 08:00:00 --mem 64G"
            echo ""
            echo "  # Dry run to see what would be created"
            echo "  $0 --dry-run"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=== Pre-group Voxel-wise Analysis Launcher ==="
echo "Output directory: $OUTPUT_DIR"
echo "Work directory: $WORKDIR"
echo "Script directory: $SCRIPT_DIR"
echo "Derivatives directory: $DERIVATIVES_DIR"
echo "Data source: $DATA_SOURCE"

# Build command for script generator
CMD="python3 create_pre_group_voxelWise.py"
CMD="$CMD --script-dir '$SCRIPT_DIR'"
CMD="$CMD --derivatives-dir '$DERIVATIVES_DIR'"

# Add workdir if specified
if [[ -n "$WORKDIR" ]]; then
    CMD="$CMD --workdir '$WORKDIR'"
fi

# Only add output-dir if it's different from default
if [[ "$OUTPUT_DIR" != "/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/groupLevel" ]]; then
    CMD="$CMD --output-dir '$OUTPUT_DIR'"
fi

if [[ -n "$SUBJECTS" ]]; then
    CMD="$CMD --subjects '$SUBJECTS'"
fi

if [[ -n "$PHASES" ]]; then
    CMD="$CMD --phases '$PHASES'"
fi

if [[ -n "$DATA_SOURCE" ]]; then
    CMD="$CMD --data-source '$DATA_SOURCE'"
fi

if [[ -n "$TIME" ]]; then
    CMD="$CMD --time '$TIME'"
fi

if [[ -n "$MEM" ]]; then
    CMD="$CMD --mem '$MEM'"
fi

if [[ -n "$DRY_RUN" ]]; then
    CMD="$CMD $DRY_RUN"
fi

echo ""
echo "Generating SLURM scripts..."
echo "Command: $CMD"
echo ""

# Generate SLURM scripts
eval $CMD

if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ SLURM scripts generated successfully!"
    
    if [[ -z "$DRY_RUN" ]]; then
        echo ""
        echo "🚀 Launching jobs..."
        
        # Change to script directory
        cd "$SCRIPT_DIR"
        
        # Launch all jobs
        bash launch_all_pre_group.sh
        
        echo ""
        echo "📊 To monitor jobs:"
        echo "  cd $SCRIPT_DIR"
        echo "  bash monitor_jobs.sh"
        echo ""
        echo "🔍 Or check individual job status:"
        echo "  squeue -u \$USER --name='pre_group_*'"
    fi
else
    echo ""
    echo "❌ Failed to generate SLURM scripts"
    exit 1
fi
