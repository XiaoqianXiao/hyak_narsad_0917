#!/usr/bin/env python3
"""
Script to generate SLURM scripts for LSS-based first-level GLM analysis.

This script automatically detects all trials from behavioral data and generates
SLURM scripts for running LSS (Least Squares Separate) analysis on each trial.
It uses the first_level_wf_LSS workflow from first_level_workflows.py.

Usage:
    # Generate SLURM scripts for all subjects and tasks
    python create_1st_LSS.py
    
    # Generate scripts for specific subjects only
    python create_1st_LSS.py --subjects N101 N102 N103
    
    # Generate scripts for specific tasks only
    python create_1st_LSS.py --tasks phase2 phase3
    
    # Generate scripts with custom SLURM settings
    python create_1st_LSS.py --account psych --partition cpu-g2-mem2x --memory 64G
    
    # Show help
    python create_1st_LSS.py --help

Features:
    - Auto-detects all trials from behavioral CSV files
    - Generates SLURM scripts for each trial
    - Uses first_level_wf_LSS workflow from first_level_workflows.py
    - Configurable SLURM resource requirements
    - Supports multiple subjects and tasks
    - Handles special case files (e.g., N202 phase3)
    - Creates organized directory structure

Output:
    - SLURM scripts in /gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss/{task}/
    - Script naming: sub_{subject}_trial_{trial_ID}_slurm.sh
    - Each script runs run_LSS.py with appropriate parameters
"""

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import os
import sys
import argparse
import json
import pandas as pd
from pathlib import Path
from bids.layout import BIDSLayout
from first_level_workflows import first_level_wf_LSS
from nipype import config, logging

# Set FSL environment
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['FSLDIR'] = '/usr/local/fsl'
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'bin')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Plugin settings
PLUGIN_SETTINGS = {
    'plugin': 'MultiProc',
    'plugin_args': {'n_procs': 4, 'raise_insufficient': False, 'maxtasksperchild': 1}
}

# Nipype configuration
config.set('execution', 'remove_unnecessary_outputs', 'false')
logging.update_logging(config)

# =============================================================================
# PATHS AND DIRECTORIES
# =============================================================================

# Base paths
ROOT_DIR = os.getenv('DATA_DIR', '/data')
PROJECT_NAME = 'NARSAD'
DATA_DIR = os.path.join(ROOT_DIR, PROJECT_NAME, 'MRI')
DERIVATIVES_DIR = os.path.join(DATA_DIR, 'derivatives')
BEHAV_DIR = os.path.join(DATA_DIR, 'source_data', 'behav')
OUTPUT_DIR = os.path.join(DERIVATIVES_DIR, 'fMRI_analysis', 'LSS')

# Container and workflow paths
SCRUBBED_DIR = '/scrubbed_dir'
CONTAINER_PATH = "/gscratch/scrubbed/fanglab/xiaoqian/images/narsad-fmri_1st_level_1.0.sif"

# BIDS configuration
SPACE = ['MNI152NLin2009cAsym']

# =============================================================================
# SLURM SCRIPT GENERATION
# =============================================================================

def create_slurm_script(subject, task, trial_ID, work_dir, slurm_config):
    """
    Create a SLURM script for LSS analysis of a specific trial.
    
    Args:
        subject (str): Subject ID
        task (str): Task name
        trial_ID (int): Trial ID
        work_dir (str): Working directory for the script
        slurm_config (dict): SLURM configuration parameters
    
    Returns:
        str: Path to the created SLURM script
    """
    # SLURM script template
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=LSS_{subject}_{task}_trial{trial_ID}
#SBATCH --account={slurm_config['account']}
#SBATCH --partition={slurm_config['partition']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={slurm_config['cpus_per_task']}
#SBATCH --mem={slurm_config['memory']}
#SBATCH --time={slurm_config['time']}
#SBATCH --output={work_dir}/sub_{subject}_trial_{trial_ID}_%j.out
#SBATCH --error={work_dir}/sub_{subject}_trial_{trial_ID}_%j.err

# Load required modules
module load apptainer

# Run LSS analysis using the container
apptainer exec -B /gscratch/fang:/data -B /gscratch/scrubbed/fanglab/xiaoqian:/scrubbed_dir {CONTAINER_PATH} \\
    python3 /app/run_LSS.py \\
    --subject {subject} \\
    --task {task} \\
    --trial {trial_ID}

echo "LSS analysis completed for subject {subject}, task {task}, trial {trial_ID}"
"""
    
    # Create script file
    script_filename = f'sub_{subject}_trial_{trial_ID}_slurm.sh'
    script_path = os.path.join(work_dir, script_filename)
    
    with open(script_path, 'w') as f:
        f.write(slurm_script)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    return script_path

def get_events_file_path(subject, task):
    """
    Get the path to the events file for a specific subject and task.
    
    Args:
        subject (str): Subject ID
        task (str): Task name
    
    Returns:
        str: Path to the events file
    """
    # Handle special case for N202 phase3
    if subject == 'N202' and task == 'phase3':
        events_file = os.path.join(BEHAV_DIR, 'single_trial_task-NARSAD_phase-3_sub-202_half_events.csv')
    else:
        # Standard naming convention
        events_file = os.path.join(BEHAV_DIR, f'single_trial_task-Narsad_{task}_half_events.csv')
    
    return events_file

def process_subject_task(layout, subject, task, slurm_config, dry_run=False):
    """
    Process a specific subject-task combination and generate SLURM scripts.
    
    Args:
        layout: BIDS layout object
        subject (str): Subject ID
        task (str): Task name
        slurm_config (dict): SLURM configuration
        dry_run (bool): If True, don't create files, just show what would be done
    
    Returns:
        int: Number of scripts created
    """
    # Query for BOLD files
    query = {
        'desc': 'preproc', 
        'suffix': 'bold', 
        'extension': ['.nii', '.nii.gz'],
        'subject': subject, 
        'task': task, 
        'space': SPACE[0]
    }
    
    bold_files = layout.get(**query)
    if not bold_files:
        print(f"  No BOLD files found for sub-{subject}, task-{task}")
        return 0
    
    # Get events file path
    events_file = get_events_file_path(subject, task)
    
    # Check if events file exists
    if not os.path.exists(events_file):
        print(f"  Events file not found: {events_file}")
        return 0
    
    try:
        # Read events file
        events_df = pd.read_csv(events_file)
        
        # Get unique trial IDs
        trial_ids = events_df['trial_ID'].unique()
        print(f"  Found {len(trial_ids)} trials in events file")
        
        if dry_run:
            print(f"  [DRY RUN] Would create {len(trial_ids)} SLURM scripts")
            return len(trial_ids)
        
        # Create work directory
        work_dir = os.path.join(SCRUBBED_DIR, PROJECT_NAME, f'work_flows/Lss/{task}')
        os.makedirs(work_dir, exist_ok=True)
        
        # Generate SLURM scripts for each trial
        scripts_created = 0
        for trial_ID in trial_ids:
            try:
                script_path = create_slurm_script(subject, task, trial_ID, work_dir, slurm_config)
                print(f"    Created: {os.path.basename(script_path)}")
                scripts_created += 1
            except Exception as e:
                print(f"    Error creating script for trial {trial_ID}: {e}")
        
        return scripts_created
        
    except Exception as e:
        print(f"  Error processing events file: {e}")
        return 0

def main():
    """Main function to generate LSS SLURM scripts."""
    
    # =============================================================================
    # ARGUMENT PARSING
    # =============================================================================
    
    parser = argparse.ArgumentParser(
        description="Generate SLURM scripts for LSS-based first-level GLM analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate scripts for all subjects and tasks
    python create_1st_LSS.py
    
    # Generate scripts for specific subjects only
    python create_1st_LSS.py --subjects N101 N102 N103
    
    # Generate scripts for specific tasks only
    python create_1st_LSS.py --tasks phase2 phase3
    
    # Generate scripts with custom SLURM settings
    python create_1st_LSS.py --account psych --partition cpu-g2-mem2x --memory 64G
    
    # Show what would be created without actually creating files
    python create_1st_LSS.py --dry-run
        """
    )
    
    # Subject and task filtering
    parser.add_argument('--subjects', nargs='+', 
                       help='Specific subjects to process (default: all)')
    parser.add_argument('--tasks', nargs='+', 
                       help='Specific tasks to process (default: all)')
    
    # SLURM configuration
    parser.add_argument('--account', default='fang',
                       help='SLURM account (default: fang)')
    parser.add_argument('--partition', default='cpu-g2',
                       help='SLURM partition (default: cpu-g2)')
    parser.add_argument('--cpus-per-task', type=int, default=4,
                       help='CPUs per task (default: 4)')
    parser.add_argument('--memory', default='40G',
                       help='Memory requirement (default: 40G)')
    parser.add_argument('--time', default='2:00:00',
                       help='Time limit (default: 2:00:00)')
    
    # Other options
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be created without creating files')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # =============================================================================
    # INITIALIZATION
    # =============================================================================
    
    print("=" * 60)
    print("LSS SLURM Script Generator")
    print("=" * 60)
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be created")
        print()
    
    # Create BIDS layout
    try:
        layout = BIDSLayout(str(DATA_DIR), validate=False, derivatives=str(DERIVATIVES_DIR))
        print(f"BIDS layout created from: {DATA_DIR}")
    except Exception as e:
        print(f"Error creating BIDS layout: {e}")
        sys.exit(1)
    
    # Get subjects and tasks
    all_subjects = layout.get_subjects()
    all_tasks = layout.get_tasks()
    
    # Apply filters
    subjects_to_process = args.subjects if args.subjects else all_subjects
    tasks_to_process = args.tasks if args.tasks else all_tasks
    
    print(f"Subjects to process: {len(subjects_to_process)}")
    print(f"Tasks to process: {len(tasks_to_process)}")
    print()
    
    # SLURM configuration
    slurm_config = {
        'account': args.account,
        'partition': args.partition,
        'cpus_per_task': args.cpus_per_task,
        'memory': args.memory,
        'time': args.time
    }
    
    print("SLURM Configuration:")
    for key, value in slurm_config.items():
        print(f"  {key}: {value}")
    print()
    
    # =============================================================================
    # SCRIPT GENERATION
    # =============================================================================
    
    total_scripts = 0
    total_subjects = 0
    
    for subject in subjects_to_process:
        if subject not in all_subjects:
            print(f"Warning: Subject {subject} not found in BIDS layout, skipping")
            continue
        
        subject_scripts = 0
        print(f"Processing subject: {subject}")
        
        for task in tasks_to_process:
            if task not in all_tasks:
                print(f"  Warning: Task {task} not found in BIDS layout, skipping")
                continue
            
            print(f"  Task: {task}")
            scripts_created = process_subject_task(
                layout, subject, task, slurm_config, args.dry_run
            )
            subject_scripts += scripts_created
        
        if subject_scripts > 0:
            total_subjects += 1
            total_scripts += subject_scripts
            print(f"  Total scripts for {subject}: {subject_scripts}")
        print()
    
    # =============================================================================
    # SUMMARY
    # =============================================================================
    
    print("=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    
    if args.dry_run:
        print(f"DRY RUN: Would create {total_scripts} SLURM scripts")
        print(f"DRY RUN: Would process {total_subjects} subjects")
        print()
        print("To actually create the scripts, run without --dry-run")
    else:
        print(f"Created {total_scripts} SLURM scripts")
        print(f"Processed {total_subjects} subjects")
        print()
        print("Scripts are located in:")
        for task in tasks_to_process:
            work_dir = os.path.join(SCRUBBED_DIR, PROJECT_NAME, f'work_flows/Lss/{task}')
            print(f"  {task}: {work_dir}")
    
    print()
    print("To submit all jobs, you can use:")
    print("  for script in /gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss/*/*.sh; do sbatch \"$script\"; done")
    print()
    print("Or use the launch script:")
    print("  ./launch_group_LSS.sh")
    print("=" * 60)

if __name__ == '__main__':
    main()
