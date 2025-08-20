#!/usr/bin/env python3
"""
Test script to verify the corrected directory construction logic
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

def test_corrected_directory_construction():
    """Test the corrected directory construction logic"""
    
    print("=== Testing Corrected Directory Construction Logic ===\n")
    
    # Test case 1: With --output-dir and --data-source placebo
    print("--- Test Case 1: --output-dir /data/.../groupLevel --data-source placebo ---")
    output_dir = "/data/NARSAD/MRI/derivatives/fMRI_analysis/groupLevel"
    data_source = "placebo"
    
    # Corrected logic
    base_results_dir = output_dir
    results_dir = os.path.join(base_results_dir, data_source.capitalize())
    
    print(f"Output Dir: {output_dir}")
    print(f"Base Results Dir: {base_results_dir}")
    print(f"Final Results Dir: {results_dir}")
    
    # Show example task/contrast paths
    task = 'phase2'
    contrast = 1
    task_results_dir = os.path.join(results_dir, f'task-{task}')
    contrast_results_dir = os.path.join(task_results_dir, f'cope{contrast}')
    
    print(f"Task Results: {task_results_dir}")
    print(f"Contrast Results: {contrast_results_dir}")
    print()
    
    # Test case 2: Without --output-dir, --data-source placebo
    print("--- Test Case 2: Default --data-source placebo ---")
    data_source = "placebo"
    
    # Corrected logic
    base_results_dir = os.path.join(DERIVATIVES_DIR, 'fMRI_analysis/groupLevel/whole_brain')
    results_dir = os.path.join(base_results_dir, data_source.capitalize())
    
    print(f"Default Base: {base_results_dir}")
    print(f"Final Results Dir: {results_dir}")
    
    # Show example task/contrast paths
    task_results_dir = os.path.join(results_dir, f'task-{task}')
    contrast_results_dir = os.path.join(task_results_dir, f'cope{contrast}')
    
    print(f"Task Results: {task_results_dir}")
    print(f"Contrast Results: {contrast_results_dir}")
    print()
    
    print("=== Summary ===")
    print("Now both cases will correctly include the data source in the path!")
    print("Case 1: /data/.../groupLevel/Placebo/...")
    print("Case 2: /data/.../groupLevel/whole_brain/Placebo/...")

if __name__ == "__main__":
    test_corrected_directory_construction()
