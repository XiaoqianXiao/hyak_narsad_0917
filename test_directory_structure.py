#!/usr/bin/env python3
"""
Test script to check directory construction logic in run_pre_group_voxelWise.py
"""

import os
import sys

# Mock the environment variables and constants
os.environ['DATA_DIR'] = '/data'
os.environ['SCRUBBED_DIR'] = '/scrubbed_dir'

# Constants from run_pre_group_voxelWise.py
ROOT_DIR = '/data'
PROJECT_NAME = 'NARSAD'
DATA_DIR = os.path.join(ROOT_DIR, PROJECT_NAME, 'MRI')
DERIVATIVES_DIR = os.path.join(DATA_DIR, 'derivatives')
SCRUBBED_DIR = '/scrubbed_dir'

def test_directory_construction():
    """Test the directory construction logic"""
    
    print("=== Testing Directory Construction Logic ===\n")
    
    # Test different data source scenarios
    test_cases = [
        ('standard', 'Standard data source'),
        ('placebo', 'Placebo data source'),
        ('guess', 'Guess data source')
    ]
    
    for data_source, description in test_cases:
        print(f"--- {description} ---")
        
        # Results directory construction
        if data_source == 'standard':
            base_results_dir = os.path.join(DERIVATIVES_DIR, 'fMRI_analysis/groupLevel/whole_brain')
            results_dir = base_results_dir
        else:
            base_results_dir = os.path.join(DERIVATIVES_DIR, 'fMRI_analysis/groupLevel/whole_brain')
            results_dir = os.path.join(base_results_dir, data_source.capitalize())
        
        # Workflow directory construction
        if data_source == 'standard':
            base_workflow_dir = os.path.join(SCRUBBED_DIR, PROJECT_NAME, 'work_flows/groupLevel/whole_brain')
            workflow_dir = base_workflow_dir
        else:
            base_workflow_dir = os.path.join(SCRUBBED_DIR, PROJECT_NAME, 'work_flows/groupLevel/whole_brain')
            workflow_dir = os.path.join(base_workflow_dir, data_source.capitalize())
        
        print(f"Data Source: {data_source}")
        print(f"Results Base: {base_results_dir}")
        print(f"Results Dir:  {results_dir}")
        print(f"Workflow Base: {base_workflow_dir}")
        print(f"Workflow Dir:  {workflow_dir}")
        
        # Show example task/contrast paths
        task = 'phase2'
        contrast = 5
        task_results_dir = os.path.join(results_dir, f'task-{task}')
        contrast_results_dir = os.path.join(task_results_dir, f'cope{contrast}')
        
        print(f"Task Results: {task_results_dir}")
        print(f"Contrast Results: {contrast_results_dir}")
        print()
    
    print("=== Summary ===")
    print("Standard: groupLevel/whole_brain/")
    print("Placebo:  groupLevel/whole_brain/Placebo/")
    print("Guess:    groupLevel/whole_brain/Guess/")

if __name__ == "__main__":
    test_directory_construction()
