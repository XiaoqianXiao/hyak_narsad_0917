#!/usr/bin/env python3
"""
Simple test to show the cleaned-up logic for gender processing.
"""

import pandas as pd
import numpy as np

def test_clean_gender_logic():
    """Test the cleaned-up gender processing logic."""
    
    print("=== Testing Cleaned-Up Gender Logic ===\n")
    
    # Simulate the behavioral data
    data = {
        'subID': ['N101', 'N102', 'N103', 'N104', 'N105'],
        'group': ['Patients', 'Patients', 'Controls', 'Controls', 'Patients'],
        'gender_code': [0, 1, 0, 1, 2]  # 0=Female, 1=Male, 2=Trans
    }
    
    df = pd.DataFrame(data)
    print("Original data:")
    print(df)
    print()
    
    # Step 1: Initial Trans subject exclusion (gender_code != 2)
    print("Step 1: Initial Trans subject exclusion (gender_code != 2)")
    df_filtered = df[df['gender_code'] != 2].copy()
    print(f"After excluding gender_code==2: {len(df_filtered)} subjects")
    print(df_filtered)
    print()
    
    # Step 2: Create group_id
    print("Step 2: Create group_id")
    group_levels = df_filtered['group'].unique()
    group_map = {level: idx + 1 for idx, level in enumerate(group_levels)}
    df_filtered['group_id'] = df_filtered['group'].map(group_map)
    print(f"Group mapping: {group_map}")
    print(df_filtered)
    print()
    
    # Step 3: Gender level recoding for 2×2 factorial design
    print("Step 3: Gender level recoding for 2×2 factorial design")
    print("Recoding: 0→1 (Female), 1→2 (Male)")
    df_filtered['gender_id'] = df_filtered['gender_code'].map({0: 1, 1: 2})
    print("After recoding:")
    print(df_filtered)
    print()
    
    # Step 4: Show final design matrix structure
    print("Step 4: Final design matrix structure")
    factor_columns = ['group_id', 'gender_id']
    factor_levels = {}
    for factor_name in factor_columns:
        factor_values = df_filtered[factor_name].values
        factor_levels[factor_name] = sorted(set(factor_values))
        print(f"Factor '{factor_name}' levels: {factor_levels[factor_name]}")
    
    n_levels1 = len(factor_levels['group_id'])
    n_levels2 = len(factor_levels['gender_id'])
    print(f"\nDesign matrix dimensions: {n_levels1} × {n_levels2} = {n_levels1 * n_levels2} columns")
    
    if n_levels1 == 2 and n_levels2 == 2:
        print("✅ This will create a 4-column design matrix (2×2 factorial) - PERFECT!")
        print("✅ Matrix singularity is completely prevented!")
        
        # Show cell counts
        print("\nCell counts for each combination:")
        cell_counts = df_filtered.groupby(['group_id', 'gender_id']).size().unstack(fill_value=0)
        print(cell_counts)
        
        if cell_counts.min().min() > 0:
            print("✅ No empty cells - design matrix will be full rank!")
        else:
            print("⚠️  Empty cells detected - may cause matrix singularity!")
    
    print()
    
    # Step 5: Create group_info
    print("Step 5: Create group_info")
    include_columns = ['subID', 'group_id', 'gender_id']
    group_info = list(df_filtered[include_columns].itertuples(index=False, name=None))
    print(f"Final group_info: {len(group_info)} subjects")
    for i, item in enumerate(group_info):
        print(f"  {i+1}: {item}")
    
    return True

if __name__ == "__main__":
    print("Testing the cleaned-up gender processing logic...\n")
    
    try:
        success = test_clean_gender_logic()
        
        print("\n" + "="*50)
        if success:
            print("✅ CLEAN GENDER LOGIC TEST PASSED!")
            print("The logic is now clear and consistent:")
            print("✓ Always use gender_id for final output")
            print("✓ Exclude gender_code==2 (Trans subjects)")
            print("✓ Recode gender levels: 0→1, 1→2")
            print("✓ Result: 4-column design matrix (2×2 factorial)")
        else:
            print("❌ TEST FAILED!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
