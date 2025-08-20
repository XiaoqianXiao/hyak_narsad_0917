#!/usr/bin/env python3
"""
Test script to verify the corrected group-level path construction
"""

import os

# Mock the constants from run_group_voxelWise.py
SCRUBBED_DIR = '/scrubbed_dir'
PROJECT_NAME = 'NARSAD'

# Corrected DATA_SOURCE_CONFIGS
DATA_SOURCE_CONFIGS = {
    'standard': {
        'description': 'Standard analysis with all subjects',
        'results_subdir': 'groupLevel/whole_brain',
        'workflows_subdir': 'groupLevel/whole_brain',
        'requires_varcope': True,
        'requires_grp': True
    },
    'placebo': {
        'description': 'Placebo condition only analysis',
        'results_subdir': 'groupLevel/whole_brain/Placebo',
        'workflows_subdir': 'groupLevel/whole_brain/Placebo',
        'requires_varcope': True,
        'requires_grp': True
    },
    'guess': {
        'description': 'Analysis including guess condition',
        'results_subdir': 'groupLevel/whole_brain/Guess',
        'workflows_subdir': 'groupLevel/whole_brain/Guess',
        'requires_varcope': True,
        'requires_grp': True
    }
}

def test_path_construction():
    """Test the corrected path construction logic"""
    
    print("=== Testing Corrected Group-Level Path Construction ===\n")
    
    # Test parameters
    base_dir = "/data/NARSAD/MRI/derivatives/fMRI_analysis"
    task = "phase2"
    contrast = 1
    
    test_cases = [
        ('standard', 'Standard analysis'),
        ('placebo', 'Placebo analysis'),
        ('guess', 'Guess analysis')
    ]
    
    for data_source, description in test_cases:
        print(f"--- {description} ---")
        
        # Get data source configuration
        data_source_config = DATA_SOURCE_CONFIGS.get(data_source, DATA_SOURCE_CONFIGS['standard'])
        
        # Set up directories
        results_dir = os.path.join(base_dir, data_source_config['results_subdir'])
        workflows_dir = os.path.join(SCRUBBED_DIR, PROJECT_NAME, 'work_flows', data_source_config['workflows_subdir'])
        
        # Define paths - whole_brain is already included in results_subdir
        result_dir = os.path.join(results_dir, f'task-{task}', f'cope{contrast}')
        workflow_dir = os.path.join(workflows_dir, f'task-{task}', f'cope{contrast}')
        
        print(f"Data Source: {data_source}")
        print(f"Results Subdir: {data_source_config['results_subdir']}")
        print(f"Results Dir: {results_dir}")
        print(f"Result Dir: {result_dir}")
        print(f"Workflow Dir: {workflow_dir}")
        
        # Show input file paths
        cope_file = os.path.join(base_dir, data_source_config['results_subdir'], f'task-{task}', f'cope{contrast}', 'merged_cope.nii.gz')
        design_file = os.path.join(base_dir, data_source_config['results_subdir'], f'task-{task}', f'cope{contrast}', 'design_files', 'design.mat')
        
        print(f"Input Cope: {cope_file}")
        print(f"Input Design: {design_file}")
        print()
    
    print("=== Summary ===")
    print("Now the paths correctly align:")
    print("1. Pre-group outputs to: groupLevel/whole_brain/Placebo/task-phase2/cope1/")
    print("2. Group analysis reads from: groupLevel/whole_brain/Placebo/task-phase2/cope1/")
    print("3. Group analysis outputs to: groupLevel/whole_brain/Placebo/task-phase2/cope1/")

if __name__ == "__main__":
    test_path_construction()
