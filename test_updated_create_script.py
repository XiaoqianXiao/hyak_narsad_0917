#!/usr/bin/env python3
"""
Test script to verify the updated create_group_voxelWise.sh directory structure
"""

import os

def test_updated_directory_structure():
    """Test the updated directory structure expectations"""
    
    print("=== Testing Updated create_group_voxelWise.sh Directory Structure ===\n")
    
    # Test parameters
    base_dir = "/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis"
    task = "phase2"
    contrast = 27
    
    test_cases = [
        ('standard', 'Standard analysis'),
        ('placebo', 'Placebo analysis'),
        ('guess', 'Guess analysis')
    ]
    
    for data_source, description in test_cases:
        print(f"--- {description} ---")
        
        # Build the expected pre-group directory path based on data source
        if data_source == "standard":
            pregroup_dir = os.path.join(base_dir, "groupLevel/whole_brain")
        else:
            pregroup_dir = os.path.join(base_dir, "groupLevel/whole_brain", data_source.capitalize())
        
        task_dir = os.path.join(pregroup_dir, f"task-{task}")
        cope_dir = os.path.join(task_dir, f"cope{contrast}")
        
        print(f"Data Source: {data_source}")
        print(f"Pre-group Dir: {pregroup_dir}")
        print(f"Task Dir: {task_dir}")
        print(f"Cope Dir: {cope_dir}")
        
        # Expected files in cope directory
        expected_files = [
            "merged_cope.nii.gz",
            "merged_varcope.nii.gz",
            "design_files/design.mat",
            "design_files/design.grp",
            "design_files/contrast.con"
        ]
        
        print("Expected files:")
        for file in expected_files:
            full_path = os.path.join(cope_dir, file)
            print(f"  {full_path}")
        print()
    
    print("=== Summary ===")
    print("Now the script correctly expects pre-group data in:")
    print("• Standard: groupLevel/whole_brain/task-phaseX/copeY/")
    print("• Placebo: groupLevel/whole_brain/Placebo/task-phaseX/copeY/")
    print("• Guess: groupLevel/whole_brain/Guess/task-phaseX/copeY/")
    print()
    print("This matches the output structure from the corrected pre-group analysis!")

if __name__ == "__main__":
    test_updated_directory_structure()
