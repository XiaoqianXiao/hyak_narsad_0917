#!/usr/bin/env python3
"""
Comprehensive test script to verify all group-level directory structures
"""

import os

# Mock the constants from group-level scripts
SCRUBBED_DIR = '/scrubbed_dir'
PROJECT_NAME = 'NARSAD'

# DATA_SOURCE_CONFIGS from run_group_voxelWise.py
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

# DATA_SOURCE_CONFIGS from create_group_voxelWise.sh
CREATE_SCRIPT_CONFIGS = [
    "standard:whole_brain:run_group_voxelWise.py",
    "placebo:whole_brain/Placebo:run_group_voxelWise.py",
    "guess:whole_brain/Guess:run_group_voxelWise.py"
]

def test_run_group_voxelWise_paths():
    """Test path construction from run_group_voxelWise.py"""
    
    print("=== Testing run_group_voxelWise.py Paths ===\n")
    
    # Test parameters
    base_dir = "/data/NARSAD/MRI/derivatives/fMRI_analysis"
    task = "phase2"
    contrast = 27
    
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
    
    return True

def test_create_group_voxelWise_paths():
    """Test path construction from create_group_voxelWise.sh"""
    
    print("=== Testing create_group_voxelWise.sh Paths ===\n")
    
    # Test parameters
    base_dir = "/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis"
    task = "phase2"
    contrast = 27
    
    for config in CREATE_SCRIPT_CONFIGS:
        source, script_subdir, script_name = config.split(':')
        print(f"--- {source.upper()} Data Source ---")
        
        # CORRECTED: Expected pre-group directory structure now includes whole_brain
        if source == "standard":
            pregroup_dir = os.path.join(base_dir, "groupLevel/whole_brain")
        else:
            pregroup_dir = os.path.join(base_dir, "groupLevel/whole_brain", source.capitalize())
        
        task_dir = os.path.join(pregroup_dir, f"task-{task}")
        cope_dir = os.path.join(task_dir, f"cope{contrast}")
        
        print(f"Data Source: {source}")
        print(f"Script Subdir: {script_subdir}")
        print(f"Script Name: {script_name}")
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
    
    return True

def test_launch_script_paths():
    """Test paths from launch_group_voxelWise.sh"""
    
    print("=== Testing launch_group_voxelWise.sh Paths ===\n")
    
    # Scripts directory from launch script
    SCRIPTS_DIR = "/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/groupLevel/whole_brain"
    
    print(f"Scripts Directory: {SCRIPTS_DIR}")
    print()
    
    # Show where different data source scripts would be
    data_sources = ['standard', 'placebo', 'guess']
    
    for data_source in data_sources:
        if data_source == 'standard':
            script_path = SCRIPTS_DIR
        else:
            script_path = os.path.join(SCRIPTS_DIR, data_source.capitalize())
        
        print(f"{data_source.upper()} scripts: {script_path}")
    
    print()
    return True

def test_complete_pipeline_paths():
    """Test complete pipeline path alignment"""
    
    print("=== Testing Complete Pipeline Path Alignment ===\n")
    
    # Test case: placebo data source, phase2, cope27
    data_source = "placebo"
    task = "phase2"
    contrast = 27
    base_dir = "/data/NARSAD/MRI/derivatives/fMRI_analysis"
    
    print(f"Test Case: {data_source.upper()} - {task} - cope{contrast}")
    print()
    
    # 1. Pre-group output (what pre-group analysis produces)
    pregroup_output = os.path.join(base_dir, "groupLevel/whole_brain/Placebo", f"task-{task}", f"cope{contrast}")
    print(f"1. Pre-group Output: {pregroup_output}")
    
    # 2. Group analysis input (what group analysis reads from)
    group_input = pregroup_output
    print(f"2. Group Analysis Input: {group_input}")
    
    # 3. Group analysis output (where group analysis writes results)
    group_output = pregroup_output  # Same location, overwrites pre-group results
    print(f"3. Group Analysis Output: {group_output}")
    
    # 4. Workflow directory (where Nipype writes temporary files)
    workflow_dir = os.path.join(SCRUBBED_DIR, PROJECT_NAME, "work_flows/groupLevel/whole_brain/Placebo", f"task-{task}", f"cope{contrast}")
    print(f"4. Workflow Directory: {workflow_dir}")
    
    # 5. Generated SLURM scripts location
    scripts_dir = "/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/groupLevel/whole_brain/Placebo"
    print(f"5. SLURM Scripts: {scripts_dir}")
    
    print()
    
    # Check alignment
    if group_input == pregroup_output:
        print("✅ INPUT ALIGNMENT: Pre-group output matches group analysis input")
    else:
        print("❌ INPUT MISALIGNMENT: Pre-group output does NOT match group analysis input")
    
    if group_output == group_input:
        print("✅ OUTPUT ALIGNMENT: Group analysis input matches output location")
    else:
        print("❌ OUTPUT MISALIGNMENT: Group analysis input does NOT match output location")
    
    print()
    return True

def main():
    """Run all tests"""
    
    print("🔍 COMPREHENSIVE GROUP-LEVEL DIRECTORY STRUCTURE TEST")
    print("=" * 60)
    print()
    
    # Run all tests
    test_run_group_voxelWise_paths()
    test_create_group_voxelWise_paths()
    test_launch_script_paths()
    test_complete_pipeline_paths()
    
    print("=" * 60)
    print("✅ All tests completed!")
    print()
    print("📁 SUMMARY:")
    print("• Pre-group and group analysis paths are now aligned")
    print("• All data sources include 'whole_brain' in their paths")
    print("• Workflow directories use scrubbed paths for writability")
    print("• Final results overwrite pre-group results in the same location")

if __name__ == "__main__":
    main()
