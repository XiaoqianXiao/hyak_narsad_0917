#!/usr/bin/env python3
"""
Generic pre-group level fMRI analysis pipeline for NARSAD project.

This script prepares first-level fMRI data for group-level analysis by:
1. Collecting cope and varcope files from first-level analysis
2. Running data preparation workflows for each task and contrast
3. Organizing data for subsequent group-level statistical analysis

Supports flexible filtering and analysis configurations through command-line arguments.

Usage:
    python run_pre_group_level.py --filter-column Drug --filter-value Placebo
    python run_pre_group_level.py --filter-column drug_condition --filter-value DrugA
    python run_pre_group_level.py --filter-column group --filter-value Patients
    python run_pre_group_level.py --filter-column guess --filter-value High
    python run_pre_group_level.py --filter-column Drug --filter-value Placebo --include-columns group_id,drug_id
    python run_pre_group_level.py --filter-column Drug --filter-value Placebo --output-dir /custom/path

Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)
"""

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import os
import shutil
import logging
import argparse
from pathlib import Path
from bids.layout import BIDSLayout
import pandas as pd
from nipype import Workflow, Node
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import DataSink
from group_level_workflows import wf_data_prepare
from templateflow.api import get as tpl_get, templates as get_tpl_list

# Configure Nipype crash directory to a writable location
import nipype
import os

# Set crash directory to a writable location
os.environ['NIPYPE_CRASH_DIR'] = '/tmp/nipype_crashes'
nipype.config.set('execution', 'crashfile_format', 'txt')
nipype.config.set('execution', 'crash_dir', '/tmp/nipype_crashes')
nipype.config.set('execution', 'remove_unnecessary_outputs', 'false')
nipype.config.set('execution', 'crashfile_format', 'txt')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# ENVIRONMENT AND PATH SETUP
# =============================================================================

# Set FSL environment variables
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['FSLDIR'] = '/usr/local/fsl'  # Matches the Docker image
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'bin')

# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

# Use environment variables for data paths
ROOT_DIR = os.getenv('DATA_DIR', '/data')
PROJECT_NAME = 'NARSAD'
DATA_DIR = os.path.join(ROOT_DIR, PROJECT_NAME, 'MRI')
DERIVATIVES_DIR = os.path.join(DATA_DIR, 'derivatives')
SCRUBBED_DIR = os.getenv('SCRUBBED_DIR', '/scrubbed_dir')
CONTAINER_PATH = "/gscratch/scrubbed/fanglab/xiaoqian/images/narsad-fmri_1st_level_1.0.sif"

# Define standard reference image (MNI152 template from FSL)
GROUP_MASK = str(tpl_get('MNI152NLin2009cAsym', resolution=2, desc='brain', suffix='mask'))

# =============================================================================
# SUBJECT EXCLUSION LISTS
# =============================================================================

# Subjects without MRI data for each phase
SUBJECTS_NO_MRI = {
    'phase2': ['N102', 'N208'],
    'phase3': ['N102', 'N208', 'N120']
}

# =============================================================================
# BEHAVIORAL DATA CONFIGURATION
# =============================================================================

# Behavioral data paths
SCR_DIR = os.path.join(ROOT_DIR, PROJECT_NAME, 'EDR')
DRUG_FILE = os.path.join(SCR_DIR, 'drug_order.csv')
ECR_FILE = os.path.join(SCR_DIR, 'ECR.csv')

# Analysis parameters
TASKS = ['phase2', 'phase3']

def get_contrast_range(task):
    """
    Get dynamic contrast range based on task.
    
    Args:
        task (str): Task name ('phase2' or 'phase3')
    
    Returns:
        list: Range of contrast numbers
    """
    if task == 'phase2':
        # Phase 2: 7 conditions → 42 contrasts (7 × 6)
        return list(range(1, 43))
    elif task == 'phase3':
        # Phase 3: 6 conditions → 30 contrasts (6 × 5)
        return list(range(1, 31))
    else:
        # Default fallback
        return list(range(1, 43))

# Default contrast range (will be overridden per task)
# This is a fallback - actual ranges are determined dynamically per task
CONTRAST_RANGE = list(range(1, 43))  # Contrasts 1-42 (fallback)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_behavioral_data(filter_column=None, filter_value=None, include_columns=None):
    """
    Load and prepare behavioral data for analysis with flexible filtering.
    
    Args:
        filter_column (str): Column name to filter on (e.g., 'Drug', 'group', 'guess')
        filter_value (str): Value to filter for (e.g., 'Placebo', 'Patients', 'High')
        include_columns (list): List of columns to include in group_info
    
    Returns:
        pandas.DataFrame: Filtered behavioral data with appropriate mappings
    """
    try:
        # Load drug order data with automatic separator detection
        from utils import read_csv_with_detection
        df_drug = read_csv_with_detection(DRUG_FILE)
        df_drug['group'] = df_drug['subID'].apply(
            lambda x: 'Patients' if x.startswith('N1') else 'Controls'
        )
        
        # Load ECR data with automatic separator detection
        df_ECR = read_csv_with_detection(ECR_FILE)
        
        # Merge behavioral data
        df_behav = df_drug.merge(df_ECR, how='left', left_on='subID', right_on='subID')
        
        # Apply filtering if specified
        if filter_column and filter_value:
            if filter_column not in df_behav.columns:
                raise ValueError(f"Filter column '{filter_column}' not found in behavioral data. "
                               f"Available columns: {list(df_behav.columns)}")
            
            # Apply filter
            df_behav = df_behav[df_behav[filter_column] == filter_value]
            logger.info(f"Filtered data by {filter_column}={filter_value}: {len(df_behav)} subjects remaining")
        
        # Create ID mappings for categorical variables
        group_levels = df_behav['group'].unique()
        group_map = {level: idx + 1 for idx, level in enumerate(group_levels)}
        df_behav['group_id'] = df_behav['group'].map(group_map)
        
        # Create drug condition mapping (handle both possible column names)
        drug_column = None
        if 'drug_condition' in df_behav.columns:
            drug_column = 'drug_condition'
        elif 'Drug' in df_behav.columns:
            drug_column = 'Drug'
        
        if drug_column:
            drug_levels = df_behav[drug_column].unique()
            drug_map = {level: idx + 1 for idx, level in enumerate(drug_levels)}
            df_behav['drug_id'] = df_behav[drug_column].map(drug_map)
            logger.info(f"Drug conditions: {drug_levels.tolist()}")
        
        # Create guess mapping if column exists
        if 'guess' in df_behav.columns:
            guess_levels = df_behav['guess'].unique()
            guess_map = {level: idx + 1 for idx, level in enumerate(guess_levels)}
            df_behav['guess_id'] = df_behav['guess'].map(guess_map)
            logger.info(f"Guess conditions: {guess_levels.tolist()}")
        
        # Validate include_columns
        if include_columns:
            missing_columns = [col for col in include_columns if col not in df_behav.columns]
            if missing_columns:
                raise ValueError(f"Requested columns not found: {missing_columns}. "
                               f"Available columns: {list(df_behav.columns)}")
        else:
            # Default columns: always include subID and group_id, add others if available
            include_columns = ['subID', 'group_id']
            if 'drug_id' in df_behav.columns:
                include_columns.append('drug_id')
            if 'guess_id' in df_behav.columns:
                include_columns.append('guess_id')
        
        logger.info(f"Loaded behavioral data for {len(df_behav)} subjects")
        logger.info(f"Groups: {group_levels.tolist()}")
        logger.info(f"Columns to include: {include_columns}")
        
        return df_behav, include_columns
        
    except Exception as e:
        logger.error(f"Failed to load behavioral data: {e}")
        raise

def load_first_level_data():
    """
    Load first-level analysis data and get subject list.
    
    Returns:
        tuple: (BIDSLayout, list of subject IDs)
    """
    try:
        firstlevel_dir = os.path.join(DERIVATIVES_DIR, 'fMRI_analysis/firstLevel')
        glayout = BIDSLayout(firstlevel_dir, validate=False, config=['bids', 'derivatives'])
        sub_list = sorted(glayout.get_subjects())
        
        logger.info(f"Loaded first-level data for {len(sub_list)} subjects")
        return glayout, sub_list
        
    except Exception as e:
        logger.error(f"Failed to load first-level data: {e}")
        raise

# =============================================================================
# DATA COLLECTION FUNCTIONS
# =============================================================================

def collect_task_data(task, contrast, subject_list, glayout):
    """
    Collect cope and varcope files for a specific task and contrast.
    
    Args:
        task (str): Task name (e.g., 'phase2', 'phase3')
        contrast (int): Contrast number
        subject_list (list): List of subject IDs
        glayout (BIDSLayout): BIDS layout for first-level data
    
    Returns:
        tuple: (list of cope files, list of varcope files)
    """
    copes, varcopes = [], []
    
    for sub in subject_list:
        try:
            # Get cope file
            cope_file = glayout.get(
                subject=sub, 
                task=task, 
                desc=f'cope{contrast}',
                extension=['.nii', '.nii.gz'], 
                return_type='file'
            )
            
            # Get varcope file
            varcope_file = glayout.get(
                subject=sub, 
                task=task, 
                desc=f'varcope{contrast}',
                extension=['.nii', '.nii.gz'], 
                return_type='file'
            )
            
            if cope_file and varcope_file:
                copes.append(cope_file[0])
                varcopes.append(varcope_file[0])
            else:
                logger.warning(f"Missing files for task-{task}, sub-{sub}, cope{contrast}")
                
        except Exception as e:
            logger.error(f"Error collecting data for sub-{sub}, task-{task}, cope{contrast}: {e}")
            continue
    
    return copes, varcopes

def filter_subjects_for_task(subject_list, task, df_behav):
    """
    Filter subjects for a specific task, excluding those without MRI data.
    
    Args:
        subject_list (list): Full list of subject IDs
        task (str): Task name
        df_behav (pandas.DataFrame): Behavioral data
    
    Returns:
        pandas.DataFrame: Filtered behavioral data for the task
    """
    # Get subjects to exclude for this task
    subjects_to_exclude = SUBJECTS_NO_MRI.get(task, [])
    
    # Filter behavioral data
    filtered_df = df_behav.loc[
        df_behav['subID'].isin(subject_list) & 
        ~df_behav['subID'].isin(subjects_to_exclude)
    ]
    
    logger.info(f"Task {task}: {len(filtered_df)} subjects after filtering "
                f"(excluded {len(subjects_to_exclude)} subjects without MRI)")
    
    return filtered_df

# =============================================================================
# WORKFLOW EXECUTION FUNCTIONS
# =============================================================================

def run_data_preparation_workflow(task, contrast, group_info, copes, varcopes, 
                                 contrast_results_dir, contrast_workflow_dir, include_columns):
    """
    Run data preparation workflow for a specific task and contrast.
    
    Args:
        task (str): Task name
        contrast (int): Contrast number
        group_info (list): Group information for subjects
        copes (list): List of cope file paths
        varcopes (list): List of varcope file paths
        contrast_results_dir (str): Results directory for this contrast
        contrast_workflow_dir (str): Workflow directory for this contrast
        include_columns (list): List of columns included in group_info
    """
    try:
        # Create workflow
        prepare_wf = wf_data_prepare(
            output_dir=contrast_results_dir,
            contrast=contrast,
            name=f"data_prepare_{task}_cope{contrast}"
        )
        
        # Set workflow parameters
        prepare_wf.base_dir = contrast_workflow_dir
        prepare_wf.inputs.inputnode.in_copes = copes
        prepare_wf.inputs.inputnode.in_varcopes = varcopes
        prepare_wf.inputs.inputnode.group_info = group_info
        prepare_wf.inputs.inputnode.result_dir = contrast_results_dir
        prepare_wf.inputs.inputnode.group_mask = GROUP_MASK
        
        # Set analysis-specific parameters
        # Note: use_guess parameter removed as it's not needed for design generation
        
        logger.info(f"Running data preparation for task-{task}, contrast-{contrast}")
        prepare_wf.run(plugin='MultiProc', plugin_args={'n_procs': 4})
        logger.info(f"Completed data preparation for task-{task}, contrast-{contrast}")
        
    except Exception as e:
        logger.error(f"Failed to run data preparation workflow for task-{task}, contrast-{contrast}: {e}")
        raise

def cleanup_intermediate_directories(contrast_workflow_dir):
    """
    Clean up intermediate workflow directories to save space.
    
    Args:
        contrast_workflow_dir (str): Workflow directory to clean
    """
    intermediate_dirs = [
        'merge_copes', 'merge_varcopes', 
        'resample_copes', 'resample_varcopes'
    ]
    
    for dir_name in intermediate_dirs:
        dir_path = os.path.join(contrast_workflow_dir, dir_name)
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                logger.debug(f"Cleaned up intermediate directory: {dir_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up {dir_path}: {e}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generic pre-group level fMRI analysis pipeline for NARSAD project. Can process all subjects/phases or specific subjects/phases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter by drug condition
  python run_pre_group_voxelWise.py --filter-column Drug --filter-value Placebo
  
  # Filter by group
  python run_pre_group_voxelWise.py --filter-column group --filter-value Patients
  
  # Filter by guess condition
  python run_pre_group_voxelWise.py --filter-column guess --filter-value High
  
  # Specify which columns to include
  python run_pre_group_voxelWise.py --filter-column Drug --filter-value Placebo --include-columns group_id,drug_id
  
  # Custom output directory
  python run_pre_group_voxelWise.py --filter-column Drug --filter-value Placebo --output-dir /custom/path
  
  # Process specific subject and phase
  python run_pre_group_voxelWise.py --subject sub-001 --phase phase2
  
  # Process specific subject for all phases
  python run_pre_group_voxelWise.py --subject sub-001
  
  # Process specific phase for all subjects
  python run_pre_group_voxelWise.py --phase phase3
  
  # No filtering (all subjects, all phases)
  python run_pre_group_voxelWise.py
        """
    )
    
    parser.add_argument(
        '--filter-column',
        type=str,
        help='Column name to filter on (e.g., Drug, group, guess, drug_condition)'
    )
    
    parser.add_argument(
        '--filter-value',
        type=str,
        help='Value to filter for (e.g., Placebo, Patients, High, DrugA)'
    )
    
    parser.add_argument(
        '--include-columns',
        type=str,
        help='Comma-separated list of columns to include in group_info (default: auto-detect)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Custom output directory (overrides default)'
    )
    
    parser.add_argument(
        '--workflow-dir',
        type=str,
        help='Custom workflow directory (overrides default)'
    )
    
    parser.add_argument(
        '--subject',
        type=str,
        help='Specific subject ID to process (e.g., sub-001)'
    )
    
    parser.add_argument(
        '--phase',
        choices=['phase2', 'phase3'],
        help='Specific phase to process'
    )
    
    parser.add_argument(
        '--data-source',
        choices=['all', 'placebo', 'guess'],
        default='all',
        help='Data source to process (default: all)'
    )
    
    parser.add_argument(
        '--cope',
        type=int,
        help='Specific cope number to process (e.g., 1, 2, 3)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.filter_column and not args.filter_value:
        parser.error("--filter-column requires --filter-value")
    if args.filter_value and not args.filter_column:
        parser.error("--filter-value requires --filter-column")
    
    try:
        # Parse include_columns if provided
        include_columns = None
        if args.include_columns:
            include_columns = [col.strip() for col in args.include_columns.split(',')]
        
        # Create analysis description
        if args.filter_column and args.filter_value:
            analysis_desc = f"filtered by {args.filter_column}={args.filter_value}"
        else:
            analysis_desc = "all subjects (no filtering)"
        
        logger.info(f"Starting pre-group level analysis pipeline: {analysis_desc}")
        
        # Set up directories
        if args.output_dir:
            results_dir = args.output_dir
        else:
            results_dir = os.path.join(DERIVATIVES_DIR, 'fMRI_analysis/groupLevel')
        
        if args.workflow_dir:
            workflow_dir = args.workflow_dir
        else:
            # Use a writable location for workflow directory if SCRUBBED_DIR is read-only
            workflow_dir = os.path.join(SCRUBBED_DIR, PROJECT_NAME, 'work_flows/groupLevel')
        
        # Create workflow directory (use temporary location to avoid read-only issues)
        Path(workflow_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Workflow directory: {workflow_dir}")
        
        # Load data
        df_behav, final_include_columns = load_behavioral_data(
            args.filter_column, args.filter_value, include_columns
        )
        glayout, subject_list = load_first_level_data()
        
        if len(df_behav) == 0:
            logger.error("No subjects found after filtering. Check your filter criteria.")
            return 1
        
        # Determine which tasks to process
        if args.phase:
            tasks_to_process = [args.phase]
            logger.info(f"Processing specific phase: {args.phase}")
        else:
            tasks_to_process = TASKS
            logger.info("Processing all phases")
        
        # Process each task
        for task in tasks_to_process:
            logger.info(f"Processing task: {task}")
            
            # Filter subjects for this task
            task_group_info_df = filter_subjects_for_task(subject_list, task, df_behav)
            
            # If specific subject requested, filter to that subject
            if args.subject:
                if args.subject in task_group_info_df['subID'].values:
                    task_group_info_df = task_group_info_df[task_group_info_df['subID'] == args.subject]
                    logger.info(f"Processing single subject: {args.subject}")
                else:
                    logger.warning(f"Subject {args.subject} not found in task {task}, skipping")
                    continue
            
            group_info = list(task_group_info_df[final_include_columns].itertuples(index=False, name=None))
            expected_subjects = len(group_info)
            
            if expected_subjects == 0:
                logger.warning(f"No subjects found for task {task}, skipping")
                continue
            
            # Create task directories
            task_results_dir = os.path.join(results_dir, f'task-{task}')
            task_workflow_dir = os.path.join(workflow_dir, f'task-{task}')
            
            Path(task_results_dir).mkdir(parents=True, exist_ok=True)
            
            # Create workflow directory
            Path(task_workflow_dir).mkdir(parents=True, exist_ok=True)
            
            # Get dynamic contrast range for this task
            task_contrast_range = get_contrast_range(task)
            
            # If specific cope requested, filter to that cope only
            if args.cope:
                if args.cope in task_contrast_range:
                    task_contrast_range = [args.cope]
                    logger.info(f"Processing specific cope: {args.cope}")
                else:
                    logger.warning(f"Cope {args.cope} not found in task {task}, skipping")
                    continue
            else:
                logger.info(f"Task {task}: Processing contrasts {task_contrast_range[0]}-{task_contrast_range[-1]} (total: {len(task_contrast_range)})")
            
            # Process each contrast
            for contrast in task_contrast_range:
                logger.info(f"Processing contrast {contrast}")
                
                # Create contrast directories
                contrast_results_dir = os.path.join(task_results_dir, f'cope{contrast}')
                contrast_workflow_dir = os.path.join(task_workflow_dir, f'cope{contrast}')
                
                Path(contrast_results_dir).mkdir(parents=True, exist_ok=True)
                Path(contrast_workflow_dir).mkdir(parents=True, exist_ok=True)
                
                # Collect data for this contrast
                copes, varcopes = collect_task_data(
                    task, contrast, [info[0] for info in group_info], glayout
                )
                
                # Check if we have complete data
                if len(copes) != expected_subjects or len(varcopes) != expected_subjects:
                    logger.warning(f"Skipping contrast {contrast}: Expected {expected_subjects} subjects, "
                                  f"got copes={len(copes)}, varcopes={len(varcopes)}")
                    continue
                
                # Run data preparation workflow
                run_data_preparation_workflow(
                    task, contrast, group_info, copes, varcopes,
                    contrast_results_dir, contrast_workflow_dir, final_include_columns
                )
                

        
        logger.info(f"Pre-group level analysis pipeline completed successfully: {analysis_desc}")
        
    except Exception as e:
        logger.error(f"Pre-group level analysis pipeline failed: {e}")
        raise

if __name__ == "__main__":
    exit(main())
