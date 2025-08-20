#!/usr/bin/env python3
"""
Test script to verify the final corrected pre-group logic
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

def test_final_corrected_logic():
    """Test the final corrected pre-group logic"""
    
    print("=== Testing Final Corrected Pre-Group Logic ===\n")
    
    # Test case: With --output-dir and --data-source placebo
    print("--- Test Case: --output-dir /data/.../groupLevel --data-source placebo ---")
    output_dir = "/data/NARSAD/MRI/derivatives/fMRI_analysis/groupLevel"
    data_source = "placebo"
    
    # Corrected logic
    base_results_dir = output_dir
    results_dir = os.path.join(base_results_dir, 'whole_brain', data_source.capitalize())
    
    print(f"Output Dir: {output_dir}")
    print(f"Base Results Dir: {base_results_dir}")
    print(f"Final Results Dir: {results_dir}")
    
    # Show example task/contrast paths
    task = 'phase2'
    contrast = 27
    task_results_dir = os.path.join(results_dir, f'task-{task}')
    contrast_results_dir = os.path.join(task_results_dir, f'cope{contrast}')
    
    print(f"Task Results: {task_results_dir}")
    print(f"Contrast Results: {contrast_results_dir}")
    print()
    
    print("=== Summary ===")
    print("Now the paths correctly include 'whole_brain':")
    print(f"Expected: /data/.../groupLevel/whole_brain/Placebo/task-phase2/cope27")
    print(f"Actual:   {contrast_results_dir}")
    print()
    print("This should match what group analysis expects to read from!")

if __name__ == "__main__":
    test_final_corrected_logic()
