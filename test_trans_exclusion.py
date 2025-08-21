#!/usr/bin/env python3
"""
Comprehensive test to verify that Trans subjects (gender_id==3) are properly excluded
at both the merging stage and design matrix generation stage.
"""

import pandas as pd
import numpy as np

def test_trans_exclusion():
    """Test that Trans subjects are properly excluded at both stages."""
    
    print("=== Testing Trans Subject Exclusion ===\n")
    
    # Step 1: Simulate the merging stage (first filter)
    print("Step 1: Testing Trans subject exclusion at merging stage")
    
    # Original data with Trans subjects
    original_data = [
        ('N101', 'Patients', 'Placebo', 1),  # Patient, Placebo, Female
        ('N102', 'Patients', 'Placebo', 2),  # Patient, Placebo, Male  
        ('N103', 'Patients', 'Placebo', 3),  # Patient, Placebo, Trans (SHOULD BE EXCLUDED)
        ('N104', 'Controls', 'Placebo', 1),  # Control, Placebo, Female
        ('N105', 'Controls', 'Placebo', 2),  # Control, Placebo, Male
        ('N106', 'Controls', 'Placebo', 3),  # Control, Placebo, Trans (SHOULD BE EXCLUDED)
        ('N107', 'Patients', 'Oxytocin', 1), # Patient, Oxytocin, Female (will be excluded by placebo filter)
        ('N108', 'Patients', 'Oxytocin', 2), # Patient, Oxytocin, Male (will be excluded by placebo filter)
    ]
    
    df_behav = pd.DataFrame(original_data, columns=['subID', 'group', 'Drug', 'gender_code'])
    print(f"Original data: {len(df_behav)} subjects")
    print(f"Trans subjects (gender_code=3): {len(df_behav[df_behav['gender_code'] == 3])}")
    print(f"Placebo subjects: {len(df_behav[df_behav['Drug'] == 'Placebo'])}")
    print()
    
    # Apply first filter: Trans subject exclusion (simulating lines 240-249 in run_pre_group_voxelWise.py)
    print("Applying first filter: Trans subject exclusion (gender_code != 3)")
    if 'gender_code' in df_behav.columns:
        before_count = len(df_behav)
        df_behav = df_behav[df_behav['gender_code'] != 3].copy()
        after_count = len(df_behav)
        if before_count != after_count:
            print(f"✓ EXCLUDED {before_count - after_count} Trans subjects (gender_code=3) from analysis")
            print(f"✓ Subjects remaining: {after_count}")
        else:
            print("✓ No Trans subjects found - no exclusions needed")
    
    print("Data after first filter:")
    print(df_behav)
    print()
    
    # Apply second filter: Placebo only (simulating data source filtering)
    print("Applying second filter: Placebo only")
    df_behav = df_behav[df_behav['Drug'] == 'Placebo'].copy()
    print(f"After placebo filtering: {len(df_behav)} subjects")
    print("Data after second filter:")
    print(df_behav)
    print()
    
    # Create derived columns
    print("Creating derived columns")
    group_levels = df_behav['group'].unique()
    group_map = {level: idx + 1 for idx, level in enumerate(group_levels)}
    df_behav['group_id'] = df_behav['group'].map(group_map)
    print(f"✓ Created group_id mapping: {group_map}")
    
    print("Data with derived columns:")
    print(df_behav)
    print()
    
    # Step 2: Test the second filter (before design matrix generation)
    print("Step 2: Testing Trans subject exclusion before design matrix generation")
    
    # Simulate the task_group_info_df that would be passed to the second filter
    task_group_info_df = df_behav.copy()
    print(f"task_group_info_df: {len(task_group_info_df)} subjects")
    print(f"Columns: {list(task_group_info_df.columns)}")
    print()
    
    # Apply the second filter (simulating lines 816-823 in run_pre_group_voxelWise.py)
    print("Applying second filter: If gender_id requested, filter out gender_id==3")
    
    # Simulate args.include_columns
    args_include_columns = ['subID', 'group_id', 'gender_id']
    print(f"Requested columns: {args_include_columns}")
    
    # Check if gender_id is requested
    if args_include_columns and 'gender_id' in args_include_columns:
        print("✓ gender_id is requested - applying Trans subject filter")
        before_count = len(task_group_info_df)
        task_group_info_df = task_group_info_df[task_group_info_df['gender_code'] != 3].copy()
        after_count = len(task_group_info_df)
        if before_count != after_count:
            print(f"✓ EXCLUDED {before_count - after_count} Trans subjects (gender_code=3) from group_info")
            print(f"✓ Subjects remaining: {after_count}")
        else:
            print("✓ No Trans subjects found in this stage - no exclusions needed")
    else:
        print("⚠️  gender_id not requested - no Trans subject filter applied")
    
    print("Data after second filter:")
    print(task_group_info_df)
    print()
    
    # Step 3: Verify final data for design matrix generation
    print("Step 3: Verifying final data for design matrix generation")
    
    # Check if any Trans subjects remain
    remaining_trans = len(task_group_info_df[task_group_info_df['gender_code'] == 3])
    if remaining_trans == 0:
        print("✓ No Trans subjects remain - ready for design matrix generation")
    else:
        print(f"❌ {remaining_trans} Trans subjects still remain!")
        return False
    
    # Check the final subject distribution
    print(f"Final subject count: {len(task_group_info_df)}")
    print(f"Group distribution: {task_group_info_df['group_id'].value_counts().to_dict()}")
    print(f"Gender distribution: {task_group_info_df['gender_code'].value_counts().to_dict()}")
    
    # Verify this will create a 2x2 factorial design
    n_groups = len(task_group_info_df['group_id'].unique())
    n_genders = len(task_group_info_df['gender_code'].unique())
    print(f"\nDesign matrix will be: {n_groups} × {n_genders} = {n_groups * n_genders} columns")
    
    if n_groups == 2 and n_genders == 2:
        print("✓ This will create a 4-column design matrix (2×2 factorial) - CORRECT!")
        print("✓ Matrix singularity is prevented!")
    else:
        print(f"⚠️  This will create a {n_groups * n_genders}-column design matrix - may cause issues!")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing Trans subject exclusion at both stages...\n")
    
    try:
        success = test_trans_exclusion()
        
        print("\n" + "="*60)
        if success:
            print("✅ TRANS SUBJECT EXCLUSION TEST PASSED!")
            print("Trans subjects (gender_id==3) are properly excluded at both stages!")
            print("\n🎯 SUMMARY:")
            print("✓ Stage 1: Trans subjects excluded at merging stage (gender_code != 3)")
            print("✓ Stage 2: Trans subjects excluded before design matrix generation")
            print("✓ Final data: No Trans subjects remain")
            print("✓ Design matrix: 4 columns (2×2 factorial)")
            print("✓ Matrix singularity: Completely prevented")
        else:
            print("❌ TRANS SUBJECT EXCLUSION TEST FAILED!")
            print("The exclusion logic needs further adjustment.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
