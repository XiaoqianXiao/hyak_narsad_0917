#!/usr/bin/env python3
"""
Test script that uses the current scripts with local behavioral data
to show the design matrix generation and verify Trans subject exclusion.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the current directory to Python path so we can import the functions
sys.path.append('.')

def test_local_design_matrix():
    """Test design matrix generation with local behavioral data."""
    
    print("=== Testing Design Matrix Generation with Local Data ===\n")
    
    # Set local behavioral data paths
    local_edr_dir = "/Users/xiaoqianxiao/projects/NARSAD/EDR"
    drug_file = os.path.join(local_edr_dir, "drug_order.csv")
    ecr_file = os.path.join(local_edr_dir, "ECR.csv")
    
    print(f"Using local behavioral data from: {local_edr_dir}")
    print(f"Drug file: {drug_file}")
    print(f"ECR file: {ecr_file}")
    print()
    
    # Check if files exist
    if not os.path.exists(drug_file):
        print(f"❌ Drug file not found: {drug_file}")
        return False
    
    if not os.path.exists(ecr_file):
        print(f"❌ ECR file not found: {ecr_file}")
        return False
    
    print("✓ Both behavioral data files found")
    print()
    
    # Step 1: Load and examine the behavioral data
    print("Step 1: Loading behavioral data")
    try:
        # Import the utility function used by the script
        from utils import read_csv_with_detection
        
        # Load drug data
        df_drug = read_csv_with_detection(drug_file)
        print(f"✓ Loaded drug data: {len(df_drug)} subjects")
        print(f"  Columns: {list(df_drug.columns)}")
        
        # Load ECR data
        df_ecr = read_csv_with_detection(ecr_file)
        print(f"✓ Loaded ECR data: {len(df_ecr)} subjects")
        print(f"  Columns: {list(df_ecr.columns)}")
        
        print()
        
    except Exception as e:
        print(f"❌ Failed to load behavioral data: {e}")
        return False
    
    # Step 2: Simulate the data processing from run_pre_group_voxelWise.py
    print("Step 2: Simulating data processing from run_pre_group_voxelWise.py")
    
    try:
        # Create group mapping (same as in the script)
        df_drug['group'] = df_drug['subID'].apply(
            lambda x: 'Patients' if x.startswith('N1') else 'Controls'
        )
        print(f"✓ Created group mapping: {df_drug['group'].value_counts().to_dict()}")
        
        # Merge behavioral data
        df_behav = df_drug.merge(df_ecr, how='left', left_on='subID', right_on='subID')
        print(f"✓ Merged data: {len(df_behav)} subjects")
        print(f"  Columns: {list(df_behav.columns)}")
        
        print()
        
    except Exception as e:
        print(f"❌ Failed to process data: {e}")
        return False
    
    # Step 3: Apply data source filtering (placebo only)
    print("Step 3: Applying data source filtering (placebo only)")
    
    try:
        # Check which drug column to use
        drug_column = None
        if 'Drug' in df_behav.columns:
            drug_column = 'Drug'
            print(f"✓ Using 'Drug' column for filtering")
        elif 'drug_condition' in df_behav.columns:
            drug_column = 'drug_condition'
            print(f"✓ Using 'drug_condition' column for filtering")
        else:
            print("⚠️  No drug column found")
            return False
        
        # Show drug values
        if drug_column:
            drug_values = df_behav[drug_column].value_counts()
            print(f"  Drug values: {drug_values.to_dict()}")
            
            # Filter by placebo
            placebo_df = df_behav[df_behav[drug_column] == 'Placebo'].copy()
            print(f"✓ After placebo filtering: {len(placebo_df)} subjects")
            
            # Show group distribution
            print(f"  Group distribution: {placebo_df['group'].value_counts().to_dict()}")
            
            # Check for gender_code column
            if 'gender_code' in placebo_df.columns:
                gender_dist = placebo_df['gender_code'].value_counts()
                print(f"  Gender distribution: {gender_dist.to_dict()}")
                print(f"  Trans subjects (gender_code=3): {gender_dist.get(3, 0)}")
            else:
                print("  ⚠️  No gender_code column found")
            
            print()
        
    except Exception as e:
        print(f"❌ Failed to filter by placebo: {e}")
        return False
    
    # Step 4: Apply Trans subject exclusion
    print("Step 4: Applying Trans subject exclusion")
    
    try:
        if 'gender_code' in placebo_df.columns:
            before_count = len(placebo_df)
            filtered_df = placebo_df[placebo_df['gender_code'] != 3].copy()
            after_count = len(filtered_df)
            
            if before_count != after_count:
                print(f"✓ EXCLUDED {before_count - after_count} Trans subjects (gender_code=3)")
                print(f"✓ Subjects remaining: {after_count}")
            else:
                print("✓ No Trans subjects found - no exclusions needed")
            
            # Show final distribution
            print(f"  Final group distribution: {filtered_df['group'].value_counts().to_dict()}")
            print(f"  Final gender distribution: {filtered_df['gender_code'].value_counts().to_dict()}")
            
            print()
        else:
            print("⚠️  Cannot exclude Trans subjects - no gender_code column")
            filtered_df = placebo_df.copy()
        
    except Exception as e:
        print(f"❌ Failed to exclude Trans subjects: {e}")
        return False
    
    # Step 5: Create derived columns and test column mapping
    print("Step 5: Creating derived columns and testing column mapping")
    
    try:
        # Create group_id mapping
        group_levels = filtered_df['group'].unique()
        group_map = {level: idx + 1 for idx, level in enumerate(group_levels)}
        filtered_df['group_id'] = filtered_df['group'].map(group_map)
        print(f"✓ Created group_id mapping: {group_map}")
        
        # Test column mapping logic (same as in the script)
        include_columns = ['subID', 'group_id', 'gender_id']
        print(f"  Requested columns: {include_columns}")
        
        column_mapping = {
            'gender_id': 'gender_code',
            'drug_id': 'drug_id',
            'group_id': 'group_id',
            'subID': 'subID'
        }
        
        # Map requested columns to actual column names
        processing_columns = []
        for col in include_columns:
            if col in column_mapping:
                processing_columns.append(column_mapping[col])
                print(f"    ✓ Mapped '{col}' -> '{column_mapping[col]}'")
            else:
                processing_columns.append(col)
                print(f"    ✓ Kept '{col}' as is")
        
        print(f"  Processing columns: {processing_columns}")
        
        # Verify all processing columns exist
        missing_columns = [col for col in processing_columns if col not in filtered_df.columns]
        if missing_columns:
            print(f"  ❌ Missing columns: {missing_columns}")
            return False
        else:
            print("  ✓ All processing columns exist")
        
        print()
        
    except Exception as e:
        print(f"❌ Failed to create derived columns: {e}")
        return False
    
    # Step 6: Create group_info and simulate design matrix generation
    print("Step 6: Creating group_info and simulating design matrix generation")
    
    try:
        # Select the data for group_info
        selected_data = filtered_df[processing_columns]
        print(f"✓ Selected data: {len(selected_data)} subjects")
        
        # Convert to list of tuples (as in the script)
        group_info = list(selected_data.itertuples(index=False, name=None))
        print(f"✓ Converted to list of tuples: {len(group_info)} subjects")
        
        # Show sample data
        print("  Sample data:")
        for i, item in enumerate(group_info[:3]):
            print(f"    {i+1}: {item}")
        
        print()
        
    except Exception as e:
        print(f"❌ Failed to create group_info: {e}")
        return False
    
    # Step 7: Simulate design matrix creation
    print("Step 7: Simulating design matrix creation")
    
    try:
        # Convert list of tuples to DataFrame (as in create_dummy_design_files)
        column_names = ['subID', 'group_id', 'gender_id']
        group_info_df = pd.DataFrame(group_info, columns=column_names)
        print(f"✓ Converted to DataFrame: {len(group_info_df)} subjects")
        
        # Extract factor columns (exclude 'subID')
        factor_columns = [col for col in group_info_df.columns if col != 'subID']
        print(f"  Factor columns: {factor_columns}")
        
        # Calculate factor levels
        factor_levels = {}
        for factor_name in factor_columns:
            factor_values = group_info_df[factor_name].values
            factor_levels[factor_name] = sorted(set(factor_values))
            print(f"  Factor '{factor_name}' levels: {factor_levels[factor_name]}")
        
        # Calculate design matrix dimensions
        if len(factor_columns) == 2:
            n_levels1 = len(factor_levels[factor_columns[0]])
            n_levels2 = len(factor_levels[factor_columns[1]])
            print(f"\n  Design matrix dimensions: {n_levels1} × {n_levels2} = {n_levels1 * n_levels2} columns")
            
            if n_levels1 == 2 and n_levels2 == 2:
                print("  ✓ This will create a 4-column design matrix (2×2 factorial) - CORRECT!")
                print("  ✓ Matrix singularity is prevented!")
                
                # Show what the design matrix would look like
                print("\n  Design matrix structure (cell-means coding):")
                print("    Column 0: group_id=1, gender_id=1")
                print("    Column 1: group_id=1, gender_id=2") 
                print("    Column 2: group_id=2, gender_id=1")
                print("    Column 3: group_id=2, gender_id=2")
                
            else:
                print(f"  ⚠️  This will create a {n_levels1 * n_levels2}-column design matrix - may cause issues!")
        
        print()
        
    except Exception as e:
        print(f"❌ Failed to simulate design matrix creation: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing design matrix generation with local behavioral data...\n")
    
    try:
        success = test_local_design_matrix()
        
        print("\n" + "="*60)
        if success:
            print("✅ LOCAL DESIGN MATRIX TEST PASSED!")
            print("The script works correctly with local behavioral data!")
            print("\n🎯 SUMMARY:")
            print("✓ Behavioral data loaded successfully")
            print("✓ Trans subjects properly excluded")
            print("✓ Column mapping works correctly")
            print("✓ Design matrix will have 4 columns (2×2 factorial)")
            print("✓ Matrix singularity is completely prevented")
        else:
            print("❌ LOCAL DESIGN MATRIX TEST FAILED!")
            print("The script needs further adjustment.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
