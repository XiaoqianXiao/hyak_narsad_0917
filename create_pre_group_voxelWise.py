#!/usr/bin/env python3
"""
Create SLURM scripts for pre-group voxel-wise analysis.

This script generates individual SLURM scripts for each subject and phase,
allowing parallel processing of the pre-group level analysis.

Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)

USAGE:
    # Create scripts for all subjects and phases (uses default output directory)
    python3 create_pre_group_voxelWise.py
    
    # Create scripts for specific subjects
    python3 create_pre_group_voxelWise.py --subjects sub-001,sub-002
    
    # Create scripts for specific phases
    python3 create_pre_group_voxelWise.py --phases phase2,phase3
    
    # Create scripts for specific subjects and phases
    python3 create_pre_group_voxelWise.py --subjects sub-001,sub-002 --phases phase2
    
    # Create scripts for specific data source
    python3 create_pre_group_voxelWise.py --data-source placebo
    
    # Custom SLURM parameters
    python3 create_pre_group_voxelWise.py --time 08:00:00 --mem 64G --partition ckpt-all
    
    # Custom script directory
    python3 create_pre_group_voxelWise.py --script-dir /custom/script/path
    
    # Custom output directory
    python3 create_pre_group_voxelWise.py --output-dir /custom/path
    
    # Dry run to see what would be created
    python3 create_pre_group_voxelWise.py --dry-run
    
    # Show help
    python3 create_pre_group_voxelWise.py --help

EXAMPLES:
    # Quick start with defaults (uses default output directory)
    python3 create_pre_group_voxelWise.py
    
    # Process only Phase 2 data
    python3 create_pre_group_voxelWise.py --phases phase2
    
    # Process specific subjects with custom resources
    python3 create_pre_group_voxelWise.py --subjects sub-001,sub-002 --time 06:00:00 --mem 48G
    
    # Process placebo data only
    python3 create_pre_group_voxelWise.py --data-source placebo
    
    # Process guess data for specific subjects
    python3 create_pre_group_voxelWise.py --data-source guess --subjects sub-001,sub-002
    
    # Test with dry run first
    python3 create_pre_group_voxelWise.py --dry-run
    
    # Custom output directory
    python3 create_pre_group_voxelWise.py --output-dir /custom/path

SLURM PARAMETERS:
    --partition: SLURM partition (default: ckpt-all)
    --account: SLURM account (default: fang)
    --time: Time limit (default: 04:00:00)
    --mem: Memory limit (default: 32G)
    --cpus-per-task: CPUs per task (default: 4)
    --container: Container image (default: narsad-fmri_1st_level_1.0.sif)

OUTPUT:
    Creates SLURM scripts in script_dir:
    - pre_group_sub-XXX_phaseY.sh (individual job scripts)
    - launch_all_pre_group.sh (launch all jobs)
    - monitor_jobs.sh (monitor job progress)
    - logs/ directory for job outputs
"""

import os
import argparse
import glob
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default SLURM parameters
DEFAULT_SLURM_PARAMS = {
    'partition': 'ckpt-all',
    'account': 'fang',
    'time': '04:00:00',
    'mem': '32G',
    'cpus_per_task': 4,
    'container': 'narsad-fmri_1st_level_1.0.sif'
}

def get_subject_list(derivatives_dir):
    """Get list of subjects from derivatives directory."""
    subjects = []
    for phase in ['phase2', 'phase3']:
        phase_dir = os.path.join(derivatives_dir, 'fMRI_analysis', 'firstLevel', f'task-{phase}')
        if os.path.exists(phase_dir):
            # Look for subject directories
            for item in os.listdir(phase_dir):
                if item.startswith('sub-') and os.path.isdir(os.path.join(phase_dir, item)):
                    subjects.append((item, phase))
    return subjects

def create_slurm_script(subject, phase, output_dir, script_dir, slurm_params, data_source):
    """Create a SLURM script for a specific subject and phase."""
    
    script_name = f"pre_group_{subject}_{phase}.sh"
    script_path = os.path.join(script_dir, script_name)
    
    # Container bind mounts
    container_binds = [
        "-B /gscratch/fang:/data",
        "-B /gscratch/scrubbed/fanglab/xiaoqian:/scrubbed_dir",
        "-B /gscratch/scrubbed/fanglab/xiaoqian/repo/hyak_narsad/group_level_workflows.py:/app/group_level_workflows.py",
        "-B /gscratch/scrubbed/fanglab/xiaoqian/repo/hyak_narsad/run_pre_group_voxelWise.py:/app/run_pre_group_voxelWise.py",
        "-B /gscratch/scrubbed/fanglab/xiaoqian/repo/hyak_narsad:/app/updated"
    ]
    
    # Script content
    script_content = f"""#!/bin/bash
#SBATCH --job-name=pre_group_{subject}_{phase}
#SBATCH --partition={slurm_params['partition']}
#SBATCH --account={slurm_params['account']}
#SBATCH --time={slurm_params['time']}
#SBATCH --mem={slurm_params['mem']}
#SBATCH --cpus-per-task={slurm_params['cpus_per_task']}
#SBATCH --output=logs/pre_group_{subject}_{phase}_%j.out
#SBATCH --error=logs/pre_group_{subject}_{phase}_%j.err

# Pre-group voxel-wise analysis for {subject} - {phase}
# Generated by create_pre_group_voxelWise.py

set -e

# Load modules if needed
module load apptainer

# Set environment variables
export SCRUBBED_DIR=/scrubbed_dir
export DATA_DIR=/data

# Create output directory
mkdir -p {output_dir}

# Run the pre-group analysis for this subject and phase
apptainer exec {' '.join(container_binds)} {slurm_params['container']} \\
    python3 /app/run_pre_group_voxelWise.py \\
    --output-dir {output_dir} \\
    --subject {subject} \\
    --phase {phase} \\
    --data-source {data_source}

echo "Completed pre-group analysis for {subject} - {phase}"
"""
    
    # Write script
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    return script_path

def create_launch_script(script_dir, output_dir, slurm_params, data_source):
    """Create a launch script to submit all jobs."""
    
    launch_script_path = os.path.join(script_dir, "launch_all_pre_group.sh")
    
    launch_content = f"""#!/bin/bash
# Launch script for all pre-group voxel-wise analysis jobs
# Generated by create_pre_group_voxelWise.py

set -e

# Create logs directory
mkdir -p logs

# Submit all jobs
for script in {script_dir}/pre_group_*.sh; do
    if [[ -f "$script" ]]; then
        echo "Submitting $script"
        sbatch "$script"
        sleep 1  # Small delay between submissions
    fi
done

echo "All pre-group analysis jobs submitted!"
echo "Check job status with: squeue -u $USER"
echo "Monitor logs in: logs/"
"""
    
    with open(launch_script_path, 'w') as f:
        f.write(launch_content)
    
    os.chmod(launch_script_path, 0o755)
    
    return launch_script_path

def create_monitor_script(script_dir):
    """Create a script to monitor job progress."""
    
    monitor_script_path = os.path.join(script_dir, "monitor_jobs.sh")
    
    monitor_content = f"""#!/bin/bash
# Monitor script for pre-group voxel-wise analysis jobs
# Generated by create_pre_group_voxelWise.py

echo "=== Pre-group Analysis Job Status ==="
echo "Submitted jobs:"
squeue -u $USER --name="pre_group_*" --format="%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R"

echo ""
echo "=== Job Counts ==="
echo "Running: $(squeue -u $USER --name="pre_group_*" --state=RUNNING | wc -l)"
echo "Pending: $(squeue -u $USER --name="pre_group_*" --state=PENDING | wc -l)"
echo "Completed: $(squeue -u $USER --name="pre_group_*" --state=COMPLETED | wc -l)"
echo "Failed: $(squeue -u $USER --name="pre_group_*" --state=FAILED | wc -l)"

echo ""
echo "=== Recent Logs ==="
if [[ -d "logs" ]]; then
    ls -la logs/pre_group_*.out | tail -5
fi
"""
    
    with open(monitor_script_path, 'w') as f:
        f.write(monitor_content)
    
    os.chmod(monitor_script_path, 0o755)
    
    return monitor_script_path

def main():
    """Main function to create SLURM scripts."""
    parser = argparse.ArgumentParser(
        description='Create SLURM scripts for pre-group voxel-wise analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create scripts for all subjects and phases (uses default output directory)
  python create_pre_group_voxelWise.py
  
  # Create scripts with custom output directory
  python create_pre_group_voxelWise.py --output-dir /custom/path
  
  # Create scripts with custom SLURM parameters
  python create_pre_group_voxelWise.py --time 08:00:00 --mem 64G
  
  # Create scripts for specific subjects only
  python create_pre_group_voxelWise.py --subjects sub-001,sub-002
        """
    )
    
    parser.add_argument(
        '--output-dir',
        default='/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/groupLevel',
        help='Output directory for pre-group analysis results (default: /gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/groupLevel)'
    )
    
    parser.add_argument(
        '--script-dir',
        help='Directory to save SLURM scripts (default: auto-generated based on output-dir)'
    )
    
    parser.add_argument(
        '--data-source',
        choices=['all', 'placebo', 'guess'],
        default='all',
        help='Data source to process (default: all)'
    )
    
    parser.add_argument(
        '--derivatives-dir',
        default='/gscratch/fang/NARSAD/MRI/derivatives',
        help='Path to derivatives directory (default: /gscratch/fang/NARSAD/MRI/derivatives)'
    )
    
    parser.add_argument(
        '--subjects',
        help='Comma-separated list of specific subjects to process (e.g., sub-001,sub-002)'
    )
    
    parser.add_argument(
        '--phases',
        help='Comma-separated list of phases to process (e.g., phase2,phase3)'
    )
    
    # SLURM parameters
    parser.add_argument(
        '--partition',
        default=DEFAULT_SLURM_PARAMS['partition'],
        help=f'SLURM partition (default: {DEFAULT_SLURM_PARAMS["partition"]})'
    )
    
    parser.add_argument(
        '--account',
        default=DEFAULT_SLURM_PARAMS['account'],
        help=f'SLURM account (default: {DEFAULT_SLURM_PARAMS["account"]})'
    )
    
    parser.add_argument(
        '--time',
        default=DEFAULT_SLURM_PARAMS['time'],
        help=f'SLURM time limit (default: {DEFAULT_SLURM_PARAMS["time"]})'
    )
    
    parser.add_argument(
        '--mem',
        default=DEFAULT_SLURM_PARAMS['mem'],
        help=f'SLURM memory limit (default: {DEFAULT_SLURM_PARAMS["mem"]})'
    )
    
    parser.add_argument(
        '--cpus-per-task',
        type=int,
        default=DEFAULT_SLURM_PARAMS['cpus_per_task'],
        help=f'SLURM CPUs per task (default: {DEFAULT_SLURM_PARAMS["cpus_per_task"]})'
    )
    
    parser.add_argument(
        '--container',
        default=DEFAULT_SLURM_PARAMS['container'],
        help=f'Container image (default: {DEFAULT_SLURM_PARAMS["container"]})'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be created without writing files'
    )
    
    args = parser.parse_args()
    
    # Set script directory
    if args.script_dir:
        script_dir = Path(args.script_dir)
    else:
        # Auto-generate script directory based on output directory
        script_dir = Path(args.output_dir).parent / 'slurm_scripts' / 'pre_group'
    
    if not args.dry_run:
        script_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created script directory: {script_dir}")
    
    # Get SLURM parameters
    slurm_params = {
        'partition': args.partition,
        'account': args.account,
        'time': args.time,
        'mem': args.mem,
        'cpus_per_task': args.cpus_per_task,
        'container': args.container
    }
    
    # Get subjects and phases to process
    if args.subjects:
        subjects_to_process = [s.strip() for s in args.subjects.split(',')]
        phases_to_process = ['phase2', 'phase3'] if not args.phases else [p.strip() for p in args.phases.split(',')]
        
        # Create subject-phase pairs
        subject_phase_pairs = []
        for subject in subjects_to_process:
            for phase in phases_to_process:
                subject_phase_pairs.append((subject, phase))
    else:
        # Get all subjects from derivatives directory
        logger.info(f"Scanning derivatives directory: {args.derivatives_dir}")
        subject_phase_pairs = get_subject_list(args.derivatives_dir)
        
        if args.phases:
            phases_to_process = [p.strip() for p in args.phases.split(',')]
            subject_phase_pairs = [(s, p) for s, p in subject_phase_pairs if p in phases_to_process]
    
    logger.info(f"Found {len(subject_phase_pairs)} subject-phase combinations to process")
    
    if args.dry_run:
        logger.info("DRY RUN - Would create the following scripts:")
        for subject, phase in subject_phase_pairs:
            logger.info(f"  pre_group_{subject}_{phase}.sh")
        logger.info("  launch_all_pre_group.sh")
        logger.info("  monitor_jobs.sh")
        return
    
    # Create individual SLURM scripts
    created_scripts = []
    for subject, phase in subject_phase_pairs:
        script_path = create_slurm_script(subject, phase, args.output_dir, script_dir, slurm_params, args.data_source)
        created_scripts.append(script_path)
        logger.info(f"Created: {script_path}")
    
    # Create launch script
    launch_script = create_launch_script(script_dir, args.output_dir, slurm_params, args.data_source)
    logger.info(f"Created: {launch_script}")
    
    # Create monitor script
    monitor_script = create_monitor_script(script_dir)
    logger.info(f"Created: {monitor_script}")
    
    logger.info(f"\n✅ Successfully created {len(created_scripts)} SLURM scripts!")
    logger.info(f"📁 Scripts saved to: {script_dir}")
    logger.info(f"🚀 To launch all jobs: bash {launch_script}")
    logger.info(f"📊 To monitor jobs: bash {monitor_script}")
    logger.info(f"🔍 Individual scripts can also be run separately")

if __name__ == "__main__":
    main()
