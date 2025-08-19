#!/usr/bin/env python3
"""
Create SLURM scripts for pre-group voxel-wise analysis.

This script generates individual SLURM scripts for each subject and phase,
allowing parallel processing of the pre-group level analysis.

Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)

USAGE:
    # Create scripts for all subjects and phases (uses preset defaults)
    python3 create_pre_group_voxelWise.py
    
    # Create scripts for specific subjects
    python3 create_pre_group_voxelWise.py --subjects sub-001,sub-002
    
    # Create scripts for specific phases
    python3 create_pre_group_voxelWise.py --phases phase2,phase3
    
    # Create scripts for specific subjects and phases
    python3 create_pre_group_voxelWise.py --subjects sub-001,sub-002 --phases phase2
    
    # Create scripts for specific data source
    python3 create_pre_group_voxelWise.py --data-source placebo
    
    # Dry run to see what would be created
    python3 create_pre_group_voxelWise.py --dry-run
    
    # Show help
    python3 create_pre_group_voxelWise.py --help

EXAMPLES:
    # Quick start with preset defaults
    python3 create_pre_group_voxelWise.py
    
    # Process only Phase 2 data
    python3 create_pre_group_voxelWise.py --phases phase2
    
    # Process specific subjects
    python3 create_pre_group_voxelWise.py --subjects sub-001,sub-002
    
    # Process placebo data only
    python3 create_pre_group_voxelWise.py --data-source placebo
    
    # Process guess data for specific subjects
    python3 create_pre_group_voxelWise.py --data-source guess --subjects sub-001,sub-002
    
    # Test with dry run first
    python3 create_pre_group_voxelWise.py --dry-run

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
    # The derivatives_dir should point to the fMRI_analysis directory
    # so we just need to append 'firstLevel'
    first_level_dir = os.path.join(derivatives_dir, 'firstLevel')
    
    logger.info(f"Looking for first level directory at: {first_level_dir}")
    
    if not os.path.exists(first_level_dir):
        logger.warning(f"First level directory not found: {first_level_dir}")
        return subjects
    
    # Look for subject directories (e.g., sub-N101, sub-N102, etc.)
    for item in os.listdir(first_level_dir):
        if item.startswith('sub-') and os.path.isdir(os.path.join(first_level_dir, item)):
            subject_dir = os.path.join(first_level_dir, item)
            
            # Check for session directories (e.g., ses-pilot3mm, ses-001, etc.)
            for session in os.listdir(subject_dir):
                if session.startswith('ses-') and os.path.isdir(os.path.join(subject_dir, session)):
                    session_dir = os.path.join(subject_dir, session)
                    func_dir = os.path.join(session_dir, 'func')
                    
                    if os.path.exists(func_dir):
                        # Check what phases this subject has by looking at the func files
                        phase_files = {}
                        logger.info(f"Scanning func directory: {func_dir}")
                        for file in os.listdir(func_dir):
                            if file.endswith('_bold.nii') and 'task-phase' in file:
                                # Extract phase from filename (e.g., task-phase2, task-phase3)
                                # Handle complex filenames like: sub-N101_ses-pilot3mm_task-phase3_space-MNI152NLin2009cAsym_desc-varcope9_bold.nii
                                logger.info(f"Found bold file: {file}")
                                if 'task-phase2' in file:
                                    phase_files['phase2'] = True
                                    logger.info(f"  -> Phase 2 detected")
                                elif 'task-phase3' in file:
                                    phase_files['phase3'] = True
                                    logger.info(f"  -> Phase 3 detected")
                        
                        # Add subject-phase combinations
                        for phase in phase_files.keys():
                            subjects.append((item, phase))
                        # Continue to check other sessions (don't break)
    
    logger.info(f"Found subjects: {[f'{s[0]}-{s[1]}' for s in subjects]}")
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
    
    # Convert host path to container path for output_dir
    # Replace /gscratch/fang with /data for container paths
    container_output_dir = output_dir.replace('/gscratch/fang', '/data')
    
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

# Create output directory on host (before container launch)
mkdir -p {output_dir}

# Run the pre-group analysis for this subject and phase
apptainer exec {' '.join(container_binds)} {slurm_params['container']} \\
    python3 /app/run_pre_group_voxelWise.py \\
    --output-dir {container_output_dir} \\
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
        '--data-source',
        choices=['all', 'placebo', 'guess'],
        default='all',
        help='Data source to process (default: all)'
    )
    
    parser.add_argument(
        '--subjects',
        help='Comma-separated list of specific subjects to process (e.g., sub-001,sub-002)'
    )
    
    parser.add_argument(
        '--phases',
        help='Comma-separated list of phases to process (e.g., phase2,phase3)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be created without writing files'
    )
    
    args = parser.parse_args()
    
    # Check if we're running in a container and adjust paths if needed
    container_env = os.getenv('CONTAINER', 'false')
    if container_env == 'true' or os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv'):
        logger.info("Detected container environment")
        # In container, prefer /tmp for script generation if output dir is read-only
        if not os.access(os.path.dirname(output_dir), os.W_OK):
            logger.warning("Output directory parent is not writable, will use /tmp for scripts")
    
    # Use container paths directly since this script runs inside the container
    logger.info("Using container paths directly")
    output_dir = '/data/NARSAD/MRI/derivatives/fMRI_analysis/groupLevel'
    derivatives_dir = '/data/NARSAD/MRI/derivatives/fMRI_analysis'
    
    # Set script directory - use default workdir/pregroup structure
    scrubbed_dir = os.getenv('SCRUBBED_DIR', '/scrubbed_dir')
    workdir = Path(scrubbed_dir) / 'NARSAD' / 'work_flows' / 'groupLevel'
    script_dir = workdir / 'pregroup'
    
    # Ensure script directory is absolute and in a writable location
    if not script_dir.is_absolute():
        script_dir = script_dir.resolve()
    
    logger.info(f"Script directory: {script_dir}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Output directory: {output_dir}")
    
    if not args.dry_run:
        try:
            script_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created script directory: {script_dir}")
        except OSError as e:
            if "Read-only file system" in str(e) or "Permission denied" in str(e):
                # Try multiple fallback locations
                fallback_locations = [
                    Path("/tmp") / "narsad_slurm_scripts" / "pregroup",
                    Path("/tmp") / "nipype_slurm_scripts" / "pregroup",
                    Path("/scrubbed_dir") / "temp_slurm_scripts" / "pregroup"
                ]
                
                for fallback_dir in fallback_locations:
                    try:
                        fallback_dir.mkdir(parents=True, exist_ok=True)
                        script_dir = fallback_dir
                        logger.warning(f"Target directory read-only, using fallback: {script_dir}")
                        break
                    except OSError:
                        continue
                else:
                    # If all fallbacks fail, use current directory with a unique name
                    import uuid
                    script_dir = Path.cwd() / f"slurm_scripts_{uuid.uuid4().hex[:8]}" / "pregroup"
                    script_dir.mkdir(parents=True, exist_ok=True)
                    logger.warning(f"All fallbacks failed, using current directory: {script_dir}")
            else:
                raise
    
    # Get SLURM parameters - use default values
    slurm_params = {
        'partition': DEFAULT_SLURM_PARAMS['partition'],
        'account': DEFAULT_SLURM_PARAMS['account'],
        'time': DEFAULT_SLURM_PARAMS['time'],
        'mem': DEFAULT_SLURM_PARAMS['mem'],
        'cpus_per_task': DEFAULT_SLURM_PARAMS['cpus_per_task'],
        'container': DEFAULT_SLURM_PARAMS['container']
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
        logger.info(f"Scanning derivatives directory: {derivatives_dir}")
        logger.info(f"Derivatives directory type: {type(derivatives_dir)}")
        logger.info(f"Derivatives directory absolute: {os.path.abspath(derivatives_dir)}")
        subject_phase_pairs = get_subject_list(derivatives_dir)
        
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
        script_path = create_slurm_script(subject, phase, output_dir, script_dir, slurm_params, args.data_source)
        created_scripts.append(script_path)
        logger.info(f"Created: {script_path}")
    
    # Create launch script
    launch_script = create_launch_script(script_dir, output_dir, slurm_params, args.data_source)
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
