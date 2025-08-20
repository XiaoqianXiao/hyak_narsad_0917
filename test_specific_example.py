#!/usr/bin/env python3
"""
Test script to show specific directory examples
"""

import os

# Mock the environment variables and constants
os.environ['DATA_DIR'] = '/data'
os.environ['SCRUBBED_DIR'] = '/scrubbed_dir'

# Constants from run_pre_group_voxelWise.py
ROOT_DIR = '/data'
PROJECT_NAME = 'NARSAD'
DATA_DIR = os.path.join(ROOT_DIR, PROJECT_NAME, 'MRI')
DERIVATIVES_DIR = os.path.join(DATA_DIR, 'derivatives')
SCRUBBED_DIR = '/scrubbed_dir'

def test_specific_example():
    """Test with specific parameters"""
    
    print("=== Specific Example: --data-source placebo --phase phase2 --cope 5 ===\n")
    
    # Parameters
    data_source = 'placebo'
    task = 'phase2'
    contrast = 5
    
    # Results directory construction
    base_results_dir = os.path.join(DERIVATIVES_DIR, 'fMRI_analysis/groupLevel/whole_brain')
    results_dir = os.path.join(base_results_dir, data_source.capitalize())
    
    # Workflow directory construction
    base_workflow_dir = os.path.join(SCRUBBED_DIR, PROJECT_NAME, 'work_flows/groupLevel/whole_brain')
    workflow_dir = os.path.join(base_workflow_dir, data_source.capitalize())
    
    # Task and contrast directories
    task_results_dir = os.path.join(results_dir, f'task-{task}')
    contrast_results_dir = os.path.join(task_results_dir, f'cope{contrast}')
    
    task_workflow_dir = os.path.join(workflow_dir, f'task-{task}')
    contrast_workflow_dir = os.path.join(task_workflow_dir, f'cope{contrast}')
    
    print("Parameters:")
    print(f"  Data Source: {data_source}")
    print(f"  Task: {task}")
    print(f"  Contrast: {contrast}")
    print()
    
    print("Directory Structure:")
    print(f"  Results Base:     {base_results_dir}")
    print(f"  Results Dir:      {results_dir}")
    print(f"  Task Results:     {task_results_dir}")
    print(f"  Contrast Results: {contrast_results_dir}")
    print()
    print(f"  Workflow Base:     {base_workflow_dir}")
    print(f"  Workflow Dir:      {workflow_dir}")
    print(f"  Task Workflow:     {task_workflow_dir}")
    print(f"  Contrast Workflow: {contrast_workflow_dir}")
    print()
    
    print("Final Result Directory (where files will be copied):")
    print(f"  {contrast_results_dir}")
    print()
    print("This should contain:")
    print("  - merged_cope.nii.gz")
    print("  - merged_varcope.nii.gz")
    print("  - design_files/")
    print("    - design.mat")
    print("    - design.grp")
    print("    - contrast.con")

if __name__ == "__main__":
    test_specific_example()
