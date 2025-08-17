#!/usr/bin/env python3
"""
First-level fMRI analysis pipeline for NARSAD project.

This script orchestrates the first-level fMRI analysis for individual subjects,
generating and running Nipype workflows. It supports both standard first-level
analysis and LSS (Least Squares Separate) analysis.

Usage:
    python create_1st_voxelWise.py --subject SUBJECT_ID --task TASK_NAME
    python create_1st_voxelWise.py  # Generate SLURM scripts for all subjects

Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)
"""

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import os
import json
import logging
from pathlib import Path
from bids.layout import BIDSLayout
from templateflow.api import get as tpl_get, templates as get_tpl_list
import pandas as pd
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# ENVIRONMENT AND PATH SETUP
# =============================================================================

# Set FSL environment variables for the container
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['FSLDIR'] = '/usr/local/fsl'  # Matches the Docker image
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'bin')

# Nipype plugin settings for local execution
PLUGIN_SETTINGS = {
    'plugin': 'MultiProc',
    'plugin_args': {
        'n_procs': 4,
        'raise_insufficient': False,
        'maxtasksperchild': 1,
    }
}

# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

# Use environment variables for data paths
ROOT_DIR = os.getenv('DATA_DIR', '/data')
PROJECT_NAME = 'NARSAD'
DATA_DIR = os.path.join(ROOT_DIR, PROJECT_NAME, 'MRI')
BIDS_DIR = DATA_DIR
DERIVATIVES_DIR = os.path.join(DATA_DIR, 'derivatives')
FMRIPREP_FOLDER = os.path.join(DERIVATIVES_DIR, 'fmriprep')
BEHAV_DIR = os.path.join(DATA_DIR, 'source_data/behav')
SCRUBBED_DIR = '/scrubbed_dir'
CONTAINER_PATH = "/gscratch/scrubbed/fanglab/xiaoqian/images/narsad-fmri_1st_level_1.0.sif"

# Workflow and output directories
PARTICIPANT_LABEL = []  # Can be set via args or env if needed
RUN = []
SPACE = ['MNI152NLin2009cAsym']

# Output directory
OUTPUT_DIR = os.path.join(DERIVATIVES_DIR, 'fMRI_analysis')
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# =============================================================================
# BIDS LAYOUT INITIALIZATION
# =============================================================================

def initialize_bids_layout():
    """Initialize BIDS layout and validate data availability."""
    try:
        layout = BIDSLayout(str(BIDS_DIR), validate=False, derivatives=str(DERIVATIVES_DIR))
        
        # Get available entities
        subjects = layout.get(target='subject', return_type='id')
        sessions = layout.get(target='session', return_type='id')
        runs = layout.get(target='run', return_type='id')
        
        logger.info(f"BIDS layout initialized: {len(subjects)} subjects, {len(sessions)} sessions")
        return layout, subjects, sessions, runs
        
    except Exception as e:
        logger.error(f"Failed to initialize BIDS layout: {e}")
        raise

def build_query(participant_label=None, run=None, task=None):
    """
    Build query for preprocessed BOLD files.
    
    Args:
        participant_label (list): List of participant labels to filter
        run (list): List of run numbers to filter
        task (str): Task name to filter
    
    Returns:
        dict: Query dictionary for BIDS layout
    """
    query = {
        'desc': 'preproc',
        'suffix': 'bold',
        'extension': ['.nii', '.nii.gz']
    }
    
    if participant_label:
        query['subject'] = '|'.join(participant_label)
    if run:
        query['run'] = '|'.join(run)
    if task:
        query['task'] = task
    if SPACE:
        query['space'] = '|'.join(SPACE)
    
    return query

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_condition_names_from_events(events_file):
    """
    Extract condition names from events file with proper CS- splitting.
    
    This function now uses the same logic as utils.py to create 7 conditions
    by splitting multiple CS- trials into CS-_first and CS-_others.
    
    Args:
        events_file (str): Path to events CSV file
    
    Returns:
        list: List of condition names with proper CS- splitting
    """
    try:
        if os.path.exists(events_file):
            # Use utility function for automatic separator detection
            from utils import read_csv_with_detection
            events_df = read_csv_with_detection(events_file)
            
            if 'trial_type' in events_df.columns:
                # Get raw conditions and apply CS- splitting logic
                raw_conditions = list(events_df['trial_type'].values)
                
                # Count CS- trials and create proper condition names
                cs_count = raw_conditions.count('CS-')
                if cs_count > 1:
                    # Multiple CS- trials: split into CS-_first and CS-_others
                    condition_names = ['CS-_first', 'CS-_others']
                    # Add other unique conditions (excluding ONLY the exact 'CS-' condition)
                    other_conditions = [c for c in set(raw_conditions) if c != 'CS-']
                    condition_names.extend(other_conditions)
                    logger.info(f"Split {cs_count} CS- trials into CS-_first and CS-_others. Total conditions: {len(condition_names)}")
                    logger.info(f"Raw conditions: {raw_conditions}")
                    logger.info(f"Unique conditions: {list(set(raw_conditions))}")
                    logger.info(f"Other conditions found: {other_conditions}")
                    logger.info(f"Final condition names: {condition_names}")
                else:
                    # Single or no CS- trials: use original logic
                    condition_names = sorted(events_df['trial_type'].unique().tolist())
                    logger.info(f"Using standard conditions: {len(condition_names)} total")
                
                logger.info(f"Final condition names: {condition_names}")
                return condition_names
            else:
                # If 'trial_type' not found, try alternative column names
                possible_columns = ['condition', 'event_type', 'type', 'stimulus', 'trial']
                for col in possible_columns:
                    if col in events_df.columns:
                        # Apply same CS- splitting logic to alternative columns
                        raw_conditions = list(events_df[col].values)
                        cs_count = raw_conditions.count('CS-')
                        if cs_count > 1:
                            condition_names = ['CS-_first', 'CS-_others']
                            other_conditions = [c for c in set(raw_conditions) if c != 'CS-']
                            condition_names.extend(other_conditions)
                            logger.info(f"Split {cs_count} CS- trials into CS-_first and CS-_others using column '{col}'. Total conditions: {len(condition_names)}")
                        else:
                            condition_names = sorted(events_df[col].unique().tolist())
                            logger.info(f"Using standard conditions from column '{col}': {len(condition_names)} total")
                        return condition_names
                
                # If no suitable column found, show available columns
                available_columns = list(events_df.columns)
                logger.warning(f"No 'trial_type' or alternative column found in events file: {events_file}")
                logger.warning(f"Available columns: {available_columns}")
                
                # If there are only a few columns, try to use the first non-numeric column
                for col in available_columns:
                    if not pd.api.types.is_numeric_dtype(events_df[col]):
                        # Apply same CS- splitting logic
                        raw_conditions = list(events_df[col].values)
                        cs_count = raw_conditions.count('CS-')
                        if cs_count > 1:
                            condition_names = ['CS-_first', 'CS-_others']
                            other_conditions = [c for c in set(raw_conditions) if c != 'CS-']
                            condition_names.extend(other_conditions)
                            logger.info(f"Split {cs_count} CS- trials into CS-_first and CS-_others using first non-numeric column '{col}'. Total conditions: {len(condition_names)}")
                        else:
                            condition_names = sorted(events_df[col].unique().tolist())
                            logger.info(f"Using first non-numeric column '{col}' as condition names: {condition_names}")
                        return condition_names
                
                logger.warning("No suitable column found for condition names")
        else:
            logger.warning(f"Events file does not exist: {events_file}")
    except Exception as e:
        logger.error(f"Could not read events file {events_file}: {e}")
    
    # Return empty list if events file cannot be read
    logger.warning("Returning empty condition names list")
    return []

def create_workflow_config():
    """
    Create consistent workflow configuration for all tasks.
    
    Returns:
        dict: Workflow configuration
    """
    config = {
        'use_smoothing': True,
        'fwhm': 6.0,
        'brightness_threshold': 1000,
        'high_pass_cutoff': 100,
        'use_derivatives': True,
        'model_serial_correlations': True,
        'contrast_type': 'standard'
    }
    
    logger.info(f"Created workflow configuration: {config}")
    return config

def get_events_file_path(sub, task):
    """
    Get the appropriate events file path for a subject and task.
    
    Args:
        sub (str): Subject ID
        task (str): Task name
    
    Returns:
        str: Path to events file
    """
    # Handle special case for N202 phase3
    if sub == 'N202' and task == 'phase3':
        events_file = os.path.join(BEHAV_DIR, 'task-NARSAD_phase-3_sub-202_events.csv')
    else:
        events_file = os.path.join(BEHAV_DIR, f'task-Narsad_{task}_events.csv')
    
    logger.info(f"Using events file: {events_file}")
    return events_file

def create_subject_inputs(sub, part, layout, query):
    """
    Create input dictionary for a subject.
    
    Args:
        sub (str): Subject ID
        part: BIDS entity object
        layout: BIDS layout object
        query (dict): Query dictionary
    
    Returns:
        dict: Input files dictionary for the subject
    """
    inputs = {sub: {}}
    base = {'subject', 'task'}.intersection(part.entities)
    subquery = {k: v for k, v in part.entities.items() if k in base}
    
    # Set basic inputs
    inputs[sub]['bold'] = part.path
    inputs[sub]['tr'] = part.entities['RepetitionTime']
    
    try:
        # Get mask file
        mask_files = layout.get(suffix='mask', return_type='file',
                               extension=['.nii', '.nii.gz'],
                               space=query['space'], **subquery)
        if not mask_files:
            raise IndexError("No mask files found")
        inputs[sub]['mask'] = mask_files[0]
        
        # Get regressors file
        regressor_files = layout.get(desc='confounds', return_type='file',
                                   extension=['.tsv'], **subquery)
        if not regressor_files:
            raise IndexError("No regressor files found")
        inputs[sub]['regressors'] = regressor_files[0]
        
    except IndexError as e:
        logger.error(f"Missing required file for subject {sub}: {e}")
        raise
    
    # Set events file
    task = part.entities['task']
    inputs[sub]['events'] = get_events_file_path(sub, task)
    
    logger.info(f"Created inputs for subject {sub}: {list(inputs[sub].keys())}")
    return inputs

# =============================================================================
# SLURM SCRIPT GENERATION
# =============================================================================

def create_slurm_script(sub, inputs, work_dir, output_dir, task, container_path):
    """
    Generate SLURM script for a subject.
    
    Args:
        sub (str): Subject ID
        inputs (dict): Input files dictionary
        work_dir (str): Working directory
        output_dir (str): Output directory
        task (str): Task name
        container_path (str): Path to container image
    
    Returns:
        str: Path to generated SLURM script
    """
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=first_level_sub_{sub}
#SBATCH --account=fang
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --output=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/firstLevel/{task}_sub_{sub}_%j.out
#SBATCH --error=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/firstLevel/{task}_sub_{sub}_%j.err

module load apptainer
apptainer exec \\
    -B /gscratch/fang:/data \\
    -B /gscratch/scrubbed/fanglab/xiaoqian:/scrubbed_dir \\
    -B /gscratch/scrubbed/fanglab/xiaoqian/repo/hyak_narsad/create_1st_voxelWise.py:/app/create_1st_voxelWise.py \\
    {container_path} \\
    python3 /app/create_1st_voxelWise.py --subject {sub} --task {task}
"""
    
    script_path = os.path.join(work_dir, f'sub_{sub}_slurm.sh')
    try:
        with open(script_path, 'w') as f:
            f.write(slurm_script)
        logger.info(f"SLURM script created: {script_path}")
        return script_path
    except Exception as e:
        logger.error(f"Failed to create SLURM script: {e}")
        raise

# =============================================================================
# WORKFLOW EXECUTION
# =============================================================================

def run_subject_workflow(sub, inputs, work_dir, output_dir, task):
    """
    Run first-level workflow for a single subject.
    
    Args:
        sub (str): Subject ID
        inputs (dict): Input files dictionary
        work_dir (str): Working directory
        output_dir (str): Output directory
        task (str): Task name
    """
    try:
        # Import workflows
        from first_level_workflows import first_level_wf, first_level_wf_LSS
        
        # Get workflow configuration
        config = create_workflow_config()
        
        # Get condition names from events file
        events_file = inputs[sub]['events']
        condition_names = get_condition_names_from_events(events_file)
        
        logger.info(f"Processing subject {sub}, task {task}")
        logger.info(f"Condition names: {condition_names}")
        logger.info(f"Workflow config: {config}")
        
        # Create the workflow
        workflow = first_level_wf(
            in_files=inputs,
            output_dir=output_dir,
            condition_names=condition_names,
            contrast_type=config['contrast_type'],
            fwhm=config['fwhm'],
            brightness_threshold=config['brightness_threshold'],
            high_pass_cutoff=config['high_pass_cutoff'],
            use_smoothing=config['use_smoothing'],
            use_derivatives=config['use_derivatives'],
            model_serial_correlations=config['model_serial_correlations']
        )
        
        # Set workflow base directory
        workflow.base_dir = os.path.join(work_dir, f'sub_{sub}')
        
        # Create output directory for this subject
        subject_output_dir = os.path.join(output_dir, 'firstLevel', task, f'sub-{sub}')
        Path(subject_output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Running workflow for subject {sub}, task {task}")
        logger.info(f"Workflow base directory: {workflow.base_dir}")
        logger.info(f"Output directory: {subject_output_dir}")
        
        # Run the workflow
        workflow.run(**PLUGIN_SETTINGS)
        
        logger.info(f"Workflow completed successfully for subject {sub}, task {task}")
        
    except ImportError as e:
        logger.error(f"Could not import workflows from first_level_workflows.py: {e}")
        logger.error("Make sure first_level_workflows.py is in the Python path")
        raise
    except Exception as e:
        logger.error(f"Error running workflow for subject {sub}, task {task}: {e}")
        raise

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def process_single_subject(args, layout, query):
    """
    Process a single subject with specified task.
    
    Args:
        args: Command line arguments
        layout: BIDS layout object
        query (dict): Query dictionary
    """
    found = False
    for part in layout.get(invalid_filters='allow', **query):
        if (part.entities['subject'] == args.subject and 
            (not args.task or part.entities['task'] == args.task)):
            
            found = True
            entities = part.entities
            sub = entities['subject']
            task = entities['task']
            
            # Create working directory
            work_dir = os.path.join(SCRUBBED_DIR, PROJECT_NAME, f'work_flows/firstLevel/{task}')
            Path(work_dir).mkdir(parents=True, exist_ok=True)
            
            try:
                # Create subject inputs
                inputs = create_subject_inputs(sub, part, layout, query)
                
                logger.info(f"Running first-level analysis for subject {sub}, task {task}")
                run_subject_workflow(sub, inputs, work_dir, OUTPUT_DIR, task)
                
            except Exception as e:
                logger.error(f"Failed to process subject {sub}: {e}")
                raise
            
            break
    
    if not found:
        error_msg = f"Subject {args.subject} with task {args.task} not found in preprocessed BOLD files"
        logger.error(error_msg)
        raise ValueError(error_msg)

def generate_slurm_scripts(layout, query):
    """
    Generate SLURM scripts for all subjects.
    
    Args:
        layout: BIDS layout object
        query (dict): Query dictionary
    """
    logger.info("Generating SLURM scripts for all subjects")
    
    for part in layout.get(invalid_filters='allow', **query):
        entities = part.entities
        sub = entities['subject']
        task = entities['task']
        
        # Create working directory
        work_dir = os.path.join(SCRUBBED_DIR, PROJECT_NAME, f'work_flows/firstLevel/{task}')
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # Create subject inputs
            inputs = create_subject_inputs(sub, part, layout, query)
            
            # Generate SLURM script
            script_path = create_slurm_script(sub, inputs, work_dir, OUTPUT_DIR, task, CONTAINER_PATH)
            logger.info(f"SLURM script created for subject {sub}, task {task}")
            
        except Exception as e:
            logger.error(f"Failed to generate SLURM script for subject {sub}: {e}")
            continue

def main():
    """Main execution function."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run first-level fMRI analysis.")
    parser.add_argument('--subject', type=str, help="Specific subject ID to process")
    parser.add_argument('--task', type=str, help="Specific task to process (e.g., phase2, phase3)")
    args = parser.parse_args()
    
    try:
        # Initialize BIDS layout
        layout, subjects, sessions, runs = initialize_bids_layout()
        
        # Build query
        query = build_query(PARTICIPANT_LABEL, RUN, args.task)
        
        # Validate query returns results
        prepped_bold = layout.get(**query)
        if not prepped_bold:
            logger.error(f'No preprocessed files found under: {DERIVATIVES_DIR}')
            return 1
        
        logger.info(f"Found {len(prepped_bold)} preprocessed BOLD files")
        
        if args.subject:
            # Process single subject
            process_single_subject(args, layout, query)
        else:
            # Generate SLURM scripts for all subjects
            generate_slurm_scripts(layout, query)
        
        logger.info("Processing completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
