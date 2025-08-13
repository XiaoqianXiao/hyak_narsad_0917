#!/usr/bin/env python3
"""
Script to merge LSS trial outputs into 4D NIfTI images.

This script takes the individual trial outputs from run_1st_LSS.py and merges them
into 4D NIfTI files for each subject-task combination. It's designed to work with
the LSS workflow outputs and create consolidated images for group-level analysis.

Usage:
    # Merge outputs for a specific subject and task
    python first_LSS_2_cateAlltrials.py --subject N101 --task phase2
    
    # Merge outputs for multiple subjects
    python first_LSS_2_cateAlltrials.py --subjects N101 N102 N103 --task phase2
    
    # Merge outputs for multiple tasks
    python first_LSS_2_cateAlltrials.py --subject N101 --tasks phase2 phase3
    
    # Merge multiple contrasts
    python first_LSS_2_cateAlltrials.py --subject N101 --task phase2 --contrasts 1 2 3
    
    # Show what would be created without creating files
    python first_LSS_2_cateAlltrials.py --subject N101 --task phase2 --dry-run
    
    # Show help
    python first_LSS_2_cateAlltrials.py --help

Features:
    - Automatically discovers LSS trial outputs using BIDS layout
    - Merges multiple trials into single 4D images
    - Supports multiple subjects, tasks, and contrasts
    - Handles different file naming conventions and formats
    - Creates organized output directory structure
    - Provides detailed progress feedback and error handling
    - Validates data consistency and handles shape mismatches
    - Supports dry-run mode for testing

Output:
    - 4D NIfTI files: sub-{subject}_task-{task}_contrast{contrast}.nii.gz
    - Organized by subject and task in all_subjects directory
    - Ready for group-level analysis
    - Maintains proper metadata and affine transformations
"""

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import os
import sys
import re
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from bids.layout import BIDSLayout
from nipype import config, logging

# Set FSL environment
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['FSLDIR'] = '/usr/local/fsl'
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'bin')

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

# LSS output paths
LSS_OUTPUT_DIR = os.path.join(DERIVATIVES_DIR, 'fMRI_analysis', 'LSS')
LSS_FIRST_LEVEL_DIR = os.path.join(LSS_OUTPUT_DIR, 'firstLevel')
RESULTS_DIR = os.path.join(LSS_FIRST_LEVEL_DIR, 'all_subjects')

# BIDS configuration
SPACE = 'MNI152NLin2009cAsym'

# =============================================================================
# FILE DISCOVERY AND PROCESSING
# =============================================================================

def discover_lss_outputs(layout, subject, task, contrast=1):
    """
    Discover LSS output files for a specific subject-task combination.
    
    Args:
        layout: BIDS layout object
        subject (str): Subject ID
        task (str): Task name
        contrast (int): Contrast number (default: 1)
    
    Returns:
        list: List of file paths sorted by trial number
    """
    # Query for LSS trial outputs
    query = {
        'extension': ['.nii', '.nii.gz'],
        'desc': r'trial.*',
        'suffix': 'bold',
        'subject': subject,
        'task': task,
        'space': SPACE
    }
    
    # Get all matching files
    all_files = layout.get(**query, regex_search=True)
    
    if not all_files:
        print(f"    No trial files found for subject {subject}, task {task}")
        return []
    
    # Filter for specific contrast if specified
    contrast_pattern = f'_cope{contrast}'
    filtered_files = [f for f in all_files if contrast_pattern in f.filename]
    
    if not filtered_files:
        print(f"    Warning: No files found with contrast {contrast}")
        return []
    
    # Sort files by trial number
    def extract_trial_num(file_obj):
        """Extract trial number from filename."""
        filename = file_obj.filename if hasattr(file_obj, 'filename') else str(file_obj)
        
        # Try multiple patterns for trial number extraction
        patterns = [
            r'desc-trial(\d+)',  # BIDS standard
            r'trial(\d+)',       # Alternative pattern
            r'_trial(\d+)',      # Underscore prefix
            r'trial_(\d+)'       # Underscore suffix
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return int(match.group(1))
        
        # Fallback: try to extract from path
        path_str = str(file_obj.path)
        for pattern in patterns:
            match = re.search(pattern, path_str)
            if match:
                return int(match.group(1))
        
        # Last resort: return infinity for sorting
        print(f"      Warning: Could not extract trial number from {filename}")
        return float('inf')
    
    sorted_files = sorted(filtered_files, key=extract_trial_num)
    
    # Filter out files with invalid trial numbers
    valid_files = [f for f in sorted_files if extract_trial_num(f) != float('inf')]
    
    if len(valid_files) != len(sorted_files):
        print(f"    Warning: {len(sorted_files) - len(valid_files)} files had invalid trial numbers")
    
    print(f"    Found {len(valid_files)} valid trial files for contrast {contrast}")
    return valid_files

def validate_and_prepare_data(files):
    """
    Validate and prepare data arrays for merging.
    
    Args:
        files (list): List of file objects to process
    
    Returns:
        tuple: (data_arrays, imgs, target_shape)
    """
    print(f"    Loading and validating {len(files)} trial files...")
    
    # Load all images
    imgs = []
    data_arrays = []
    valid_indices = []
    
    for i, file_obj in enumerate(files):
        try:
            file_path = file_obj.path if hasattr(file_obj, 'path') else str(file_obj)
            img = nib.load(file_path)
            
            # Basic validation
            if img is None or img.get_fdata() is None:
                print(f"      Warning: Invalid image data in file {i+1}")
                continue
            
            data = img.get_fdata()
            if data.size == 0:
                print(f"      Warning: Empty data in file {i+1}")
                continue
            
            imgs.append(img)
            data_arrays.append(data)
            valid_indices.append(i)
            
            if (i + 1) % 10 == 0:  # Progress indicator
                print(f"      Loaded trial {i+1}/{len(files)}")
                
        except Exception as e:
            print(f"      Error loading file {i+1}: {e}")
            continue
    
    if not data_arrays:
        raise ValueError("No valid data arrays loaded")
    
    # Check data consistency
    shapes = [data.shape for data in data_arrays]
    unique_shapes = set(shapes)
    
    if len(unique_shapes) > 1:
        print(f"    Warning: Inconsistent shapes found: {unique_shapes}")
        
        # Use the most common shape
        shape_counts = {}
        for shape in shapes:
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        
        target_shape = max(shape_counts, key=shape_counts.get)
        print(f"    Using most common shape: {target_shape} (appears {shape_counts[target_shape]} times)")
        
        # Filter arrays with target shape
        filtered_data = []
        filtered_imgs = []
        for i, (data, shape) in enumerate(zip(data_arrays, shapes)):
            if shape == target_shape:
                filtered_data.append(data)
                filtered_imgs.append(imgs[i])
            else:
                print(f"      Excluding trial {valid_indices[i]+1} with shape {shape}")
        
        if not filtered_data:
            raise ValueError(f"No arrays with target shape {target_shape}")
        
        data_arrays = filtered_data
        imgs = filtered_imgs
        print(f"    Proceeding with {len(data_arrays)} consistent arrays")
    else:
        target_shape = shapes[0]
        print(f"    All arrays have consistent shape: {target_shape}")
    
    return data_arrays, imgs, target_shape

def merge_trials_to_4d(files, subject, task, contrast=1):
    """
    Merge multiple trial files into a single 4D NIfTI image.
    
    Args:
        files (list): List of file paths to merge
        subject (str): Subject ID
        task (str): Task name
        contrast (int): Contrast number
    
    Returns:
        str: Path to the created 4D file
    """
    if not files:
        raise ValueError("No files provided for merging")
    
    # Validate and prepare data
    data_arrays, imgs, target_shape = validate_and_prepare_data(files)
    
    # Stack arrays along time dimension
    print(f"    Merging {len(data_arrays)} arrays into 4D image...")
    data_4d = np.stack(data_arrays, axis=-1)
    
    # Create 4D NIfTI image
    affine = imgs[0].affine
    header = imgs[0].header.copy()
    header.set_data_dtype(data_4d.dtype)
    
    # Update header information for 4D data
    header.set_data_shape(data_4d.shape)
    header.set_xyzt_units('mm', 'sec')
    
    img_4d = nib.Nifti1Image(data_4d, affine, header)
    
    # Save 4D image
    output_filename = f"sub-{subject}_task-{task}_contrast{contrast}.nii.gz"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    
    # Ensure output directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save with compression
    nib.save(img_4d, output_path)
    
    print(f"    Saved 4D NIfTI: {output_path}")
    print(f"    Final shape: {data_4d.shape}")
    print(f"    Data type: {data_4d.dtype}")
    print(f"    File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    
    return output_path

def process_subject_task(layout, subject, task, contrasts=None, dry_run=False):
    """
    Process a specific subject-task combination.
    
    Args:
        layout: BIDS layout object
        subject (str): Subject ID
        task (str): Task name
        contrasts (list): List of contrast numbers to process
        dry_run (bool): If True, don't create files, just show what would be done
    
    Returns:
        int: Number of files created
    """
    if contrasts is None:
        contrasts = [1]  # Default to contrast 1
    
    print(f"  Processing subject: {subject}, task: {task}")
    
    files_created = 0
    
    for contrast in contrasts:
        print(f"    Contrast: {contrast}")
        
        # Discover LSS outputs
        files = discover_lss_outputs(layout, subject, task, contrast)
        
        if not files:
            print(f"      No files found for contrast {contrast}")
            continue
        
        if dry_run:
            print(f"      [DRY RUN] Would merge {len(files)} files into 4D image")
            files_created += 1
            continue
        
        try:
            # Merge trials into 4D image
            output_path = merge_trials_to_4d(files, subject, task, contrast)
            files_created += 1
            
        except Exception as e:
            print(f"      Error processing contrast {contrast}: {e}")
            continue
    
    return files_created

def main():
    """Main function to merge LSS trial outputs."""
    
    # =============================================================================
    # ARGUMENT PARSING
    # =============================================================================
    
    parser = argparse.ArgumentParser(
        description="Merge LSS trial outputs into 4D NIfTI images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Merge outputs for a specific subject and task
    python first_LSS_2_cateAlltrials.py --subject N101 --task phase2
    
    # Merge outputs for multiple subjects
    python first_LSS_2_cateAlltrials.py --subjects N101 N102 N103 --task phase2
    
    # Merge outputs for multiple tasks
    python first_LSS_2_cateAlltrials.py --subject N101 --tasks phase2 phase3
    
    # Merge multiple contrasts
    python first_LSS_2_cateAlltrials.py --subject N101 --task phase2 --contrasts 1 2 3
    
    # Show what would be created without creating files
    python first_LSS_2_cateAlltrials.py --subject N101 --task phase2 --dry-run
        """
    )
    
    # Subject and task specification
    parser.add_argument('--subject', help='Specific subject to process')
    parser.add_argument('--subjects', nargs='+', help='Multiple subjects to process')
    parser.add_argument('--task', help='Specific task to process')
    parser.add_argument('--tasks', nargs='+', help='Multiple tasks to process')
    
    # Contrast specification
    parser.add_argument('--contrasts', nargs='+', type=int, default=[1],
                       help='Contrast numbers to process (default: 1)')
    
    # Other options
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be created without creating files')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # =============================================================================
    # VALIDATION
    # =============================================================================
    
    if not args.subject and not args.subjects:
        print("Error: Must specify either --subject or --subjects")
        sys.exit(1)
    
    if not args.task and not args.tasks:
        print("Error: Must specify either --task or --tasks")
        sys.exit(1)
    
    # =============================================================================
    # INITIALIZATION
    # =============================================================================
    
    print("=" * 70)
    print("LSS Trial Output Merger - Enhanced Version")
    print("=" * 70)
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be created")
        print()
    
    # Create BIDS layout
    try:
        layout = BIDSLayout(str(LSS_FIRST_LEVEL_DIR), validate=False)
        print(f"BIDS layout created from: {LSS_FIRST_LEVEL_DIR}")
    except Exception as e:
        print(f"Error creating BIDS layout: {e}")
        print(f"Please ensure the directory exists: {LSS_FIRST_LEVEL_DIR}")
        sys.exit(1)
    
    # Get subjects and tasks
    all_subjects = layout.get_subjects()
    all_tasks = layout.get_tasks()
    
    if not all_subjects:
        print(f"Warning: No subjects found in {LSS_FIRST_LEVEL_DIR}")
        print("This might indicate that LSS analysis hasn't been run yet.")
    
    if not all_tasks:
        print(f"Warning: No tasks found in {LSS_FIRST_LEVEL_DIR}")
        print("This might indicate that LSS analysis hasn't been run yet.")
    
    # Apply filters
    subjects_to_process = args.subjects if args.subjects else [args.subject]
    tasks_to_process = args.tasks if args.tasks else [args.task]
    
    print(f"Subjects to process: {subjects_to_process}")
    print(f"Tasks to process: {tasks_to_process}")
    print(f"Contrasts to process: {args.contrasts}")
    print()
    
    # =============================================================================
    # PROCESSING
    # =============================================================================
    
    total_files = 0
    total_subjects = 0
    processing_errors = []
    
    for subject in subjects_to_process:
        if subject not in all_subjects:
            print(f"Warning: Subject {subject} not found in BIDS layout, skipping")
            continue
        
        subject_files = 0
        print(f"Processing subject: {subject}")
        
        for task in tasks_to_process:
            if task not in all_tasks:
                print(f"  Warning: Task {task} not found in BIDS layout, skipping")
                continue
            
            try:
                files_created = process_subject_task(
                    layout, subject, task, args.contrasts, args.dry_run
                )
                subject_files += files_created
            except Exception as e:
                error_msg = f"Error processing subject {subject}, task {task}: {e}"
                print(f"  {error_msg}")
                processing_errors.append(error_msg)
                continue
        
        if subject_files > 0:
            total_subjects += 1
            total_files += subject_files
            print(f"  Total files for {subject}: {subject_files}")
        print()
    
    # =============================================================================
    # SUMMARY
    # =============================================================================
    
    print("=" * 70)
    print("MERGE SUMMARY")
    print("=" * 70)
    
    if args.dry_run:
        print(f"DRY RUN: Would create {total_files} 4D NIfTI files")
        print(f"DRY RUN: Would process {total_subjects} subjects")
        print()
        print("To actually create the files, run without --dry-run")
    else:
        print(f"Created {total_files} 4D NIfTI files")
        print(f"Processed {total_subjects} subjects")
        print()
        print("Output files are located in:")
        print(f"  {RESULTS_DIR}")
    
    if processing_errors:
        print()
        print("Processing errors encountered:")
        for error in processing_errors:
            print(f"  - {error}")
    
    print()
    print("The merged 4D files are ready for group-level analysis!")
    print("=" * 70)

if __name__ == '__main__':
    main()