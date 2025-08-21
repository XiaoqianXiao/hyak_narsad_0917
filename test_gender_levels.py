#!/usr/bin/env python3
"""
Test script to examine the gender levels in the local behavioral data
to understand what they represent and fix the design matrix issue.
"""

import pandas as pd
import os

def examine_gender_levels():
    """Examine the gender levels in the local behavioral data."""
    
    print("=== Examining Gender Levels in Local Data ===\n")
    
    # Set local behavioral data paths
    local_edr_dir = "/Users/xiaoqianxiao/projects/NARSAD/EDR"
    drug_file = os.path.join(local_edr_dir, "drug_order.csv")
    
    print(f"Using local behavioral data from: {local_edr_dir}")
    print(f"Drug file: {drug_file}")
    print()
    
    # Load drug data
    from utils import read_csv_with_detection
    df_drug = read_csv_with_detection(drug_file)
    
    print(f"Loaded drug data: {len(df_drug)} subjects")
    print(f"Columns: {list(df_drug.columns)}")
    print()
    
    # Examine gender-related columns
    print("Step 1: Examining gender-related columns")
    
    if 'Gender' in df_drug.columns:
        print("'Gender' column (string values):")
        gender_counts = df_drug['Gender'].value_counts()
        print(f"  Values: {gender_counts.to_dict()}")
        print()
    
    if 'gender_code' in df_drug.columns:
        print("'gender_code' column (numeric values):")
        gender_code_counts = df_drug['gender_code'].value_counts().sort_index()
        print(f"  Values: {gender_code_counts.to_dict()}")
        print()
        
        # Show the mapping between Gender and gender_code
        if 'Gender' in df_drug.columns:
            print("Mapping between 'Gender' and 'gender_code':")
            mapping = df_drug.groupby(['Gender', 'gender_code']).size().unstack(fill_value=0)
            print(mapping)
            print()
    
    # Check for any other gender-related columns
    gender_columns = [col for col in df_drug.columns if 'gender' in col.lower() or 'sex' in col.lower()]
    if gender_columns:
        print(f"Other gender-related columns: {gender_columns}")
        for col in gender_columns:
            if col not in ['Gender', 'gender_code']:
                print(f"  {col}: {df_drug[col].value_counts().to_dict()}")
        print()
    
    # Step 2: Examine the data structure
    print("Step 2: Examining data structure")
    
    # Show sample data for different gender codes
    print("Sample data for each gender_code:")
    for gender_code in sorted(df_drug['gender_code'].unique()):
        sample_data = df_drug[df_drug['gender_code'] == gender_code].head(3)
        print(f"\n  gender_code = {gender_code}:")
        for _, row in sample_data.iterrows():
            gender_str = row.get('Gender', 'N/A') if 'Gender' in df_drug.columns else 'N/A'
            print(f"    subID: {row['subID']}, Gender: {gender_str}, Drug: {row.get('Drug', 'N/A')}")
    
    print()
    
    # Step 3: Check if we need to recode gender levels
    print("Step 3: Checking if we need to recode gender levels")
    
    current_gender_levels = sorted(df_drug['gender_code'].unique())
    print(f"Current gender levels: {current_gender_levels}")
    
    if len(current_gender_levels) == 3 and 0 in current_gender_levels:
        print("⚠️  Found 3 gender levels including 0 - this will create 6-column design matrix")
        print("   Need to recode to 2 levels (1, 2) for 4-column design matrix")
        
        # Suggest recoding
        print("\n   Suggested recoding:")
        print("     gender_code 0 → recode to 1 (Female)")
        print("     gender_code 1 → recode to 2 (Male)")
        print("     gender_code 2 → exclude or recode (if not needed)")
        
        # Check what gender_code 2 represents
        if 2 in current_gender_levels:
            gender_2_samples = df_drug[df_drug['gender_code'] == 2].head(3)
            print(f"\n   gender_code 2 samples:")
            for _, row in gender_2_samples.iterrows():
                gender_str = row.get('Gender', 'N/A') if 'Gender' in df_drug.columns else 'N/A'
                print(f"     subID: {row['subID']}, Gender: {gender_str}, Drug: {row.get('Drug', 'N/A')}")
    
    print()
    
    return df_drug

if __name__ == "__main__":
    print("Examining gender levels in local behavioral data...\n")
    
    try:
        df_drug = examine_gender_levels()
        
        print("="*60)
        print("GENDER LEVEL ANALYSIS COMPLETE")
        print("="*60)
        print("Check the output above to understand the gender coding")
        print("and determine if recoding is needed for 2×2 factorial design.")
        
    except Exception as e:
        print(f"\n❌ ANALYSIS FAILED: {e}")
        import traceback
        traceback.print_exc()
