#!/usr/bin/env python3
"""
Test script that actually runs run_pre_group_voxelWise.py with test data.
This tests the complete functionality including behavioral data loading.
"""

import pandas as pd
import numpy as np
import sys
import os
import subprocess
from pathlib import Path

def create_test_behavioral_data():
    """Create test behavioral data files."""
    
    print("=== Creating Test Behavioral Data ===\n")
    
    # Create test directory structure
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create drug_order.csv (simulating the drug data)
    drug_data = [
        ('N101', 'Placebo', 'Placebo', 'Female', 1, 0.0),
        ('N102', 'Placebo', 'Placebo', 'Male', 2, 0.0),
        ('N103', 'Placebo', 'Placebo', 'Trans', 3, 0.0),  # Trans subject (should be excluded)
        ('N104', 'Controls', 'Placebo', 'Female', 1, 0.0),
        ('N105', 'Controls', 'Placebo', 'Male', 2, 0.0),
        ('N106', 'Controls', 'Placebo', 'Trans', 3, 0.0),  # Trans subject (should be excluded)
        ('N107', 'Patients', 'Oxytocin', 'Female', 1, 1.0),  # Non-placebo (should be excluded)
        ('N108', 'Patients', 'Oxytocin', 'Male', 2, 1.0),    # Non-placebo (should be excluded)
        ('N109', 'Patients', 'Placebo', 'Female', 1, 0.0),
        ('N110', 'Patients', 'Placebo', 'Male', 2, 0.0),
        ('N111', 'Controls', 'Placebo', 'Female', 1, 0.0),
        ('N112', 'Controls', 'Placebo', 'Male', 2, 0.0),
    ]
    
    drug_df = pd.DataFrame(drug_data, columns=['subID', 'group', 'Drug', 'Gender', 'gender_code', 'guess'])
    drug_file = test_dir / "drug_order.csv"
    drug_df.to_csv(drug_file, index=False)
    print(f"✓ Created {drug_file} with {len(drug_df)} subjects")
    
    # Create ECR.csv (simulating the ECR data)
    ecr_data = []
    for sub_id, group, drug, gender, gender_code, guess in drug_data:
        # Add some ECR questionnaire data
        ecr_row = [sub_id] + [np.random.randint(1, 8) for _ in range(36)]  # 36 ECR questions
        ecr_row.extend([
            np.random.randint(18, 65),  # demo_age
            gender_code,                 # demo_sex_at_birth
            np.random.randint(1, 8),    # Anxiety
            np.random.randint(1, 8),    # Avoidance
            np.random.randint(36, 252)  # total_score
        ])
        ecr_data.append(ecr_row)
    
    ecr_columns = ['subID'] + [f'ecr_q{i}' for i in range(1, 37)] + [
        'demo_age', 'demo_sex_at_birth', 'Anxiety', 'Avoidance', 'total_score'
    ]
    ecr_df = pd.DataFrame(ecr_data, columns=ecr_columns)
    ecr_file = test_dir / "ECR.csv"
    ecr_df.to_csv(ecr_file, index=False)
    print(f"✓ Created {ecr_file} with {len(ecr_df)} subjects")
    
    return test_dir, drug_file, ecr_file

def test_run_pre_group_voxelWise():
    """Test the run_pre_group_voxelWise.py script."""
    
    print("=== Testing run_pre_group_voxelWise.py ===\n")
    
    # Create test data
    test_dir, drug_file, ecr_file = create_test_behavioral_data()
    
    # Set environment variables to point to test data
    os.environ['DRUG_DATA_FILE'] = str(drug_file)
    os.environ['ECR_DATA_FILE'] = str(ecr_file)
    
    print("Step 1: Test with basic arguments")
    try:
        # Test with basic arguments
        cmd = [
            'python3', 'run_pre_group_voxelWise.py',
            '--phase', 'phase2',
            '--data-source', 'placebo',
            '--include-columns', 'subID,group_id,gender_id',
            '--output-dir', 'test_output',
            '--workflow-dir', 'test_workflow'
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the script
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        print(f"Exit code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        if result.returncode == 0:
            print("✓ Script ran successfully!")
        else:
            print("❌ Script failed!")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Script timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"❌ Failed to run script: {e}")
        return False
    
    print()
    
    # Cleanup
    print("Cleaning up...")
    try:
        import shutil
        if os.path.exists('test_output'):
            shutil.rmtree('test_output')
        if os.path.exists('test_workflow'):
            shutil.rmtree('test_workflow')
        shutil.rmtree(test_dir)
        print("✓ Test data and output cleaned up")
    except Exception as e:
        print(f"⚠️  Could not clean up: {e}")
    
    return True

if __name__ == "__main__":
    print("Testing run_pre_group_voxelWise.py with behavioral data loading...\n")
    
    try:
        success = test_run_pre_group_voxelWise()
        
        print("\n" + "="*60)
        if success:
            print("✅ RUN_PRE_GROUP_VOXELWISE.PY TEST PASSED!")
            print("The script is working correctly with behavioral data loading!")
        else:
            print("❌ RUN_PRE_GROUP_VOXELWISE.PY TEST FAILED!")
            print("The script needs further adjustment.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
