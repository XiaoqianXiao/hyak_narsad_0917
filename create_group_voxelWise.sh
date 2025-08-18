#!/bin/bash

# =============================================================================
# Unified Group-Level Voxel-Wise SLURM Script Generator for NARSAD Project
# =============================================================================
#
# This script generates SLURM scripts for group-level voxel-wise fMRI analysis.
# It supports different analysis types, data sources, and configurations.
#
# Usage:
#   ./create_group_voxelWise.sh --data-source standard
#   ./create_group_voxelWise.sh --data-source placebo
#   ./create_group_voxelWise.sh --data-source guess
#   ./create_group_voxelWise.sh --data-source standard --analysis-type flameo
#   ./create_group_voxelWise.sh --data-source placebo --account fang --partition ckpt-all
#
# Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)
#
# =============================================================================

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default settings
DEFAULT_ACCOUNT="fang"
DEFAULT_PARTITION="ckpt-all"
DEFAULT_CPUS_PER_TASK=4
DEFAULT_MEMORY="16G"
DEFAULT_TIME="8:00:00"
DEFAULT_ANALYSIS_TYPES=("randomise" "flameo")

# Tasks and contrasts
TASKS=("phase2" "phase3")

# Dynamic contrast ranges based on task
# Phase 2: 8 conditions → 56 contrasts (8 × 7)
# Phase 3: 6 conditions → 30 contrasts (6 × 5)
get_contrast_range() {
    local task="$1"
    case "$task" in
        "phase2")
            echo "$(seq 1 56)"  # 8 conditions → 56 contrasts
            ;;
        "phase3")
            echo "$(seq 1 30)"  # 6 conditions → 30 contrasts
            ;;
        *)
            echo "$(seq 1 42)"  # Default fallback
            ;;
    esac
}

# =============================================================================
# CONTAINER PATH
# =============================================================================

# Single container for all data sources
CONTAINER_PATH="/gscratch/scrubbed/fanglab/xiaoqian/images/narsad-fmri_1st_level_1.0.sif"

# =============================================================================
# DATA SOURCE CONFIGURATIONS
# =============================================================================

DATA_SOURCE_CONFIGS=(
    "standard:groupLevel/whole_brain:run_group_voxelWise.py"
    "placebo:groupLevel/whole_brain/Placebo:run_group_voxelWise.py"
    "guess:groupLevel/whole_brain:run_group_voxelWise.py"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Generate SLURM scripts for group-level voxel-wise fMRI analysis.

OPTIONS:
    --data-source TYPE     Data source type: standard, placebo, or guess (default: standard)
    --analysis-type TYPE   Analysis type: randomise, flameo, or both (default: both)
    --account ACCOUNT     SLURM account (default: $DEFAULT_ACCOUNT)
    --partition PARTITION SLURM partition (default: $DEFAULT_PARTITION)
    --cpus-per-task N     CPUs per task (default: $DEFAULT_CPUS_PER_TASK)
    --memory MEMORY       Memory requirement (default: $DEFAULT_MEMORY)
    --time TIME           Time limit (default: $DEFAULT_TIME)
    --base-dir DIR        Base directory for data (default: /gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows)
    --script-dir DIR      Directory to save SLURM scripts (default: auto-generated)
    --help                Show this help message

EXAMPLES:
    # Generate scripts for standard analysis (both randomise and flameo)
    $0 --data-source standard
    
    # Generate scripts for placebo analysis (flameo only)
    $0 --data-source placebo --analysis-type flameo
    
    # Generate scripts for guess analysis with custom settings
    $0 --data-source guess --account fang --partition ckpt-all --memory 32G
    
    # Generate scripts for standard analysis with custom base directory
    $0 --data-source standard --base-dir /custom/path

EOF
}



get_data_source_config() {
    local data_source="$1"
    for config in "${DATA_SOURCE_CONFIGS[@]}"; do
        IFS=':' read -r source script_subdir script_name <<< "$config"
        if [[ "$source" == "$data_source" ]]; then
            echo "$script_subdir:$script_name"
            return 0
        fi
    done
    echo "Error: Unknown data source: $data_source" >&2
    return 1
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

# Initialize variables with defaults
DATA_SOURCE="standard"
ANALYSIS_TYPES=("${DEFAULT_ANALYSIS_TYPES[@]}")
ACCOUNT="$DEFAULT_ACCOUNT"
PARTITION="$DEFAULT_PARTITION"
CPUS_PER_TASK="$DEFAULT_CPUS_PER_TASK"
MEMORY="$DEFAULT_MEMORY"
TIME="$DEFAULT_TIME"
BASE_DIR="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows"
SCRIPT_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-source)
            DATA_SOURCE="$2"
            shift 2
            ;;
        --analysis-type)
            if [[ "$2" == "both" ]]; then
                ANALYSIS_TYPES=("randomise" "flameo")
            else
                ANALYSIS_TYPES=("$2")
            fi
            shift 2
            ;;
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --cpus-per-task)
            CPUS_PER_TASK="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --script-dir)
            SCRIPT_DIR="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1" >&2
            show_usage
            exit 1
            ;;
    esac
done

# =============================================================================
# VALIDATION
# =============================================================================



# Validate analysis types
for analysis_type in "${ANALYSIS_TYPES[@]}"; do
    if [[ "$analysis_type" != "randomise" && "$analysis_type" != "flameo" ]]; then
        echo "Error: Invalid analysis type: $analysis_type" >&2
        echo "Valid types: randomise, flameo" >&2
        exit 1
    fi
done

# =============================================================================
# SCRIPT GENERATION
# =============================================================================

echo "=========================================="
echo "Group-Level Voxel-Wise SLURM Script Generator"
echo "=========================================="
echo "Data source: $DATA_SOURCE"
echo "Analysis types: ${ANALYSIS_TYPES[*]}"
echo "Account: $ACCOUNT"
echo "Partition: $PARTITION"
echo "CPUs per task: $CPUS_PER_TASK"
echo "Memory: $MEMORY"
echo "Time limit: $TIME"
echo "Base directory: $BASE_DIR"
echo "=========================================="

# Get data source configuration

DATA_SOURCE_CONFIG=$(get_data_source_config "$DATA_SOURCE")
if [[ $? -ne 0 ]]; then
    exit 1
fi

IFS=':' read -r SCRIPT_SUBDIR SCRIPT_NAME <<< "$DATA_SOURCE_CONFIG"

# Set script directory
if [[ -z "$SCRIPT_DIR" ]]; then
    SCRIPT_DIR="${BASE_DIR}/${SCRIPT_SUBDIR}"
fi

# Create script directory
mkdir -p "$SCRIPT_DIR"
echo "Creating SLURM scripts in: $SCRIPT_DIR"

# Generate SLURM scripts
SCRIPT_COUNT=0
for task in "${TASKS[@]}"; do
    # Get dynamic contrast range for this task
    CONTRASTS=$(get_contrast_range "$task")
    echo "Generating scripts for task: $task (contrasts: $CONTRASTS)"
    
    for contrast in $CONTRASTS; do
        for analysis_type in "${ANALYSIS_TYPES[@]}"; do
            # Create job name
            job_name="group_${task}_cope${contrast}_${analysis_type}"
            script_path="${SCRIPT_DIR}/${job_name}.sh"
            out_path="${SCRIPT_DIR}/${job_name}_%j.out"
            err_path="${SCRIPT_DIR}/${job_name}_%j.err"
            
            # Generate SLURM script content
            cat << EOF > "$script_path"
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME}
#SBATCH --output=${out_path}
#SBATCH --error=${err_path}

module load apptainer
apptainer exec -B /gscratch/fang:/data -B /gscratch/scrubbed/fanglab/xiaoqian:/scrubbed_dir ${CONTAINER_PATH} \\
    python3 /app/${SCRIPT_NAME} \\
    --task ${task} \\
    --contrast ${contrast} \\
    --analysis-type ${analysis_type} \\
    --data-source ${DATA_SOURCE} \\
    --base-dir ${BASE_DIR}

EOF
            
            # Make script executable
            chmod +x "$script_path"
            echo "Created SLURM script: $script_path"
            ((SCRIPT_COUNT++))
        done
    done
done

echo "=========================================="
echo "Voxel-wise SLURM script generation completed!"
echo "Total scripts created: $SCRIPT_COUNT"
echo "Scripts location: $SCRIPT_DIR"
echo "=========================================="
echo ""
echo "To submit all jobs, you can use:"
echo "  for script in $SCRIPT_DIR/*.sh; do sbatch \$script; done"
echo ""
echo "Or use the launch script:"
echo "  ./launch_group_level.sh"
echo "=========================================="
