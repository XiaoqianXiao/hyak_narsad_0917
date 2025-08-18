# Pre-Group Voxel-Wise Analysis Pipeline

This document describes the pre-group voxel-wise analysis pipeline for the NARSAD project, which has been optimized for parallel processing using SLURM.

## Overview

The pre-group voxel-wise analysis pipeline processes first-level fMRI results to prepare data for group-level statistical analysis. The pipeline has been redesigned to:

1. **Process individual subjects/phases** instead of all data at once
2. **Generate SLURM scripts** for parallel processing
3. **Reduce processing time** by distributing work across compute nodes
4. **Maintain data integrity** with proper error handling and logging

## Scripts

### 1. `run_pre_group_voxelWise.py` (Modified)
The main analysis script, now enhanced with:
- `--subject`: Process specific subject (e.g., `sub-001`)
- `--phase`: Process specific phase (e.g., `phase2`, `phase3`)
- Single subject/phase processing for faster execution
- Better error handling and crash file management

### 2. `create_pre_group_voxelWise.py` (New)
Generates individual SLURM scripts for each subject-phase combination:
- Auto-discovers subjects and phases from derivatives directory
- Configurable SLURM parameters (partition, account, time, memory)
- Creates launch and monitoring scripts
- Supports filtering by specific subjects or phases

### 3. `launch_pre_group_voxelWise.sh` (New)
Convenience script that:
- Generates SLURM scripts
- Launches all jobs automatically
- Provides monitoring commands

## Usage

### Quick Start (All Subjects/Phases)

```bash
# Generate and launch SLURM scripts for all subjects and phases
bash launch_pre_group_voxelWise.sh

# Or manually:
python3 create_pre_group_voxelWise.py --output-dir /gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/groupLevel
cd slurm_scripts/pre_group
bash launch_all_pre_group.sh
```

### Specific Subjects or Phases

```bash
# Process specific subjects
bash launch_pre_group_voxelWise.sh --subjects sub-001,sub-002

# Process specific phases
bash launch_pre_group_voxelWise.sh --phases phase2

# Process specific subjects for specific phases
bash launch_pre_group_voxelWise.sh --subjects sub-001,sub-002 --phases phase2,phase3
```

### Data Source Filtering

```bash
# Process all data sources (default)
bash launch_pre_group_voxelWise.sh

# Process only placebo data
bash launch_pre_group_voxelWise.sh --data-source placebo

# Process only guess data
bash launch_pre_group_voxelWise.sh --data-source guess

# Process specific data source for specific subjects
bash launch_pre_group_voxelWise.sh --data-source placebo --subjects sub-001,sub-002
```

### Custom SLURM Parameters

```bash
# Custom time and memory limits
bash launch_pre_group_voxelWise.sh --time 08:00:00 --mem 64G

# Custom work directory
bash launch_pre_group_voxelWise.sh --workdir /custom/workdir

# Custom script directory
bash launch_pre_group_voxelWise.sh --script-dir /custom/script/path
```

### Manual Processing (Single Subject/Phase)

```bash
# Process single subject and phase directly
python3 run_pre_group_voxelWise.py \
    --output-dir /gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/groupLevel \
    --subject sub-001 \
    --phase phase2

# Process single subject for all phases
python3 run_pre_group_voxelWise.py \
    --output-dir /gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/groupLevel \
    --subject sub-001

# Process single subject with specific data source
python3 run_pre_group_voxelWise.py \
    --output-dir /gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/groupLevel \
    --subject sub-001 \
    --data-source placebo
```

## SLURM Script Structure

Generated scripts are organized as follows:

```
$workdir/pregroup/
├── pre_group_sub-001_phase2.sh      # Individual job script
├── pre_group_sub-001_phase3.sh      # Individual job script
├── pre_group_sub-002_phase2.sh      # Individual job script
├── pre_group_sub-002_phase3.sh      # Individual job script
├── launch_all_pre_group.sh          # Launch all jobs
├── monitor_jobs.sh                  # Monitor job progress
└── logs/                            # Job output logs
    ├── pre_group_sub-001_phase2_*.out
    ├── pre_group_sub-001_phase2_*.err
    └── ...
```

## Job Monitoring

### Check Job Status
```bash
# View all pre-group jobs
squeue -u $USER --name="pre_group_*"

# Check job counts
cd $workdir/pregroup
bash monitor_jobs.sh
```

### View Logs
```bash
# Check recent logs
cd $workdir/pregroup/logs
ls -la pre_group_*.out | tail -5

# View specific job log
cat pre_group_sub-001_phase2_12345.out
cat pre_group_sub-001_phase2_12345.err
```

## Configuration

### Default SLURM Parameters
- **Partition**: `ckpt-all`
- **Account**: `fang`
- **Time Limit**: `04:00:00`
- **Memory**: `32G`
- **CPUs**: `4`
- **Container**: `narsad-fmri_1st_level_1.0.sif`

### Container Bind Mounts
- `/gscratch/fang:/data` - Data directory
- `/gscratch/scrubbed/fanglab/xiaoqian:/scrubbed_dir` - Scrub directory
- `/gscratch/scrubbed/fanglab/xiaoqian/repo/hyak_narsad:/app/updated` - Updated code

### Container Environment Handling
The script automatically detects when running in a container and handles read-only filesystem issues:
- **Automatic fallback**: If target directories are read-only, scripts are saved to writable locations
- **Fallback locations**: `/tmp`, `/scrubbed_dir`, or current directory with unique names
- **Path resolution**: Ensures all paths are absolute and container-compatible

## Output Structure

The pipeline generates output in the following structure:

```
/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/groupLevel/
├── task-phase2/
│   ├── cope1/
│   │   ├── design.mat
│   │   ├── design.con
│   │   ├── design.grp
│   │   ├── merged_copes.nii.gz
│   │   └── merged_varcopes.nii.gz
│   ├── cope2/
│   └── ...
└── task-phase3/
    ├── cope1/
    ├── cope2/
    └── ...
```

## Error Handling

### Crash Files
- Crash files are saved to `/tmp/nipype_crashes` to avoid read-only filesystem issues
- Nipype configuration prevents cleanup issues

### Common Issues
1. **Missing subjects**: Check if subject exists in derivatives directory
2. **Missing phases**: Verify phase directories exist
3. **SLURM failures**: Check job logs in `logs/` directory
4. **Container issues**: Verify container path and bind mounts
5. **Read-only filesystem**: Script automatically falls back to writable locations (`/tmp`, `/scrubbed_dir`)

## Performance Benefits

### Before (Sequential)
- Processing time: ~2-4 hours for all subjects/phases
- Single point of failure
- Difficult to monitor progress
- Resource underutilization

### After (Parallel)
- Processing time: ~15-30 minutes per subject/phase
- Parallel execution across compute nodes
- Easy progress monitoring
- Better resource utilization
- Fault tolerance (failed jobs don't affect others)

## Troubleshooting

### Job Stuck in PENDING
```bash
# Check partition availability
sinfo -p ckpt-all

# Check account status
sacctmgr show user $USER
```

### Job Failed
```bash
# Check error logs
cd slurm_scripts/pre_group/logs
cat pre_group_*_*_*.err

# Check container logs
scontrol show job <job_id>
```

### Missing Output
```bash
# Verify output directory permissions
ls -la /gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/groupLevel/

# Check if container bind mounts are correct
apptainer exec --bind /gscratch/fang:/data narsad-fmri_1st_level_1.0.sif ls /data
```

## Support

For issues or questions, contact:
- **Author**: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)
- **Project**: NARSAD fMRI Analysis Pipeline

## Version History

- **v2.0**: Added SLURM parallelization and single subject/phase processing
- **v1.0**: Original sequential processing pipeline
