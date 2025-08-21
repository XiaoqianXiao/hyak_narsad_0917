#!/usr/bin/env python3
"""
Test script to simulate placebo data source filtering and design matrix generation.
This will help us understand why the design matrix becomes singular.
"""

import pandas as pd
import numpy as np
import os

def load_behavioral_data():
    """Load the behavioral data from the EDR directory."""
    drug_file = "/Users/xiaoqianxiao/projects/NARSAD/EDR/drug_order.csv"
    ecr_file = "/Users/xiaoqianxiao/projects/NARSAD/EDR/ECR.csv"
    
    # Load drug order data
    df_drug = pd.read_csv(drug_file)
    print(f"Loaded drug order data: {len(df_drug)} subjects")
    print(f"Drug conditions: {df_drug['Drug'].value_counts().to_dict()}")
    print(f"Gender distribution: {df_drug['Gender'].value_counts().to_dict()}")
    
    # Load ECR data to get group information
    df_ecr = pd.read_csv(ecr_file)
    print(f"Loaded ECR data: {len(df_ecr)} subjects")
    
    # Merge the data
    df_merged = pd.merge(df_drug, df_ecr[['subID', 'demo_sex_at_birth']], on='subID', how='left')
    
    # Create group based on subject ID (same logic as run_pre_group_voxelWise.py)
    df_merged['group'] = df_merged['subID'].apply(
        lambda x: 'Patients' if x.startswith('N1') else 'Controls'
    )
    
    # Create group_id mapping
    group_levels = df_merged['group'].unique()
    group_map = {level: idx + 1 for idx, level in enumerate(group_levels)}
    df_merged['group_id'] = df_merged['group'].map(group_map)
    
    print(f"Group distribution: {df_merged['group'].value_counts().to_dict()}")
    print(f"Group mapping: {group_map}")
    
    # Create gender_id based on Gender column, but exclude Trans (gender_id=3)
    df_merged['gender_id'] = df_merged['Gender'].map({'Female': 1, 'Male': 2, 'Trans': 3})
    
    # EXCLUDE gender_id == 3 (Trans) from all future analyses
    df_merged = df_merged[df_merged['gender_id'] != 3].copy()
    print(f"EXCLUDED Trans subjects (gender_id=3) from analysis")
    
    print(f"Merged data (excluding Trans): {len(df_merged)} subjects")
    print(f"Group ID distribution: {df_merged['group_id'].value_counts().to_dict()}")
    print(f"Gender ID distribution: {df_merged['gender_id'].value_counts().to_dict()}")
    
    return df_merged

def filter_placebo_subjects(df):
    """Filter for placebo subjects only."""
    df_placebo = df[df['Drug'] == 'Placebo'].copy()
    print(f"\nAfter placebo filtering: {len(df_placebo)} subjects")
    print(f"Placebo group distribution: {df_placebo['group_id'].value_counts().to_dict()}")
    print(f"Placebo gender distribution: {df_placebo['gender_id'].value_counts().to_dict()}")
    
    # Show the actual distribution across factorial combinations
    print("\nDetailed factorial distribution:")
    factorial_dist = df_placebo.groupby(['group_id', 'gender_id']).size().reset_index(name='count')
    print(factorial_dist)
    
    # Check if factors are confounded
    if len(factorial_dist) < 4:  # Less than 2×2=4 combinations
        print("\n⚠️  WARNING: Factors are confounded! Not all factorial combinations exist.")
        print("This will cause matrix singularity regardless of coding method.")
    
    return df_placebo

def create_design_matrix(df, columns):
    """Create design matrix using the same logic as create_dummy_design_files."""
    print(f"\nCreating design matrix with columns: {columns}")
    
    # Extract factor columns (exclude 'subID' if present)
    factor_columns = [col for col in columns if col != 'subID']
    n_factors = len(factor_columns)
    
    print(f"Number of factors: {n_factors}")
    
    # Get unique levels for each factor
    factor_levels = {}
    for factor_name in factor_columns:
        factor_values = df[factor_name].values
        factor_levels[factor_name] = sorted(set(factor_values))
        print(f"Factor '{factor_name}' levels: {factor_levels[factor_name]}")
    
    # Create design matrix using cell-means coding
    if n_factors == 2:
        # Two factors (e.g., 2x2 factorial)
        factor_names = list(factor_levels.keys())
        factor1_name = factor_names[0]
        factor2_name = factor_names[1]
        
        n_levels1 = len(factor_levels[factor1_name])
        n_levels2 = len(factor_levels[factor2_name])
        
        print(f"Design matrix dimensions: {len(df)} subjects × {n_levels1 * n_levels2} cells")
        
        # Create design matrix
        design_matrix = []
        for _, row in df.iterrows():
            design_row = [0] * (n_levels1 * n_levels2)
            factor1_value = row[factor1_name]
            factor2_value = row[factor2_name]
            
            # Find the cell index
            level1_idx = factor_levels[factor1_name].index(factor1_value)
            level2_idx = factor_levels[factor2_name].index(factor2_value)
            cell_idx = level1_idx * n_levels2 + level2_idx
            
            design_row[cell_idx] = 1
            design_matrix.append(design_row)
        
        design_matrix = np.array(design_matrix)
        
        print(f"Design matrix shape: {design_matrix.shape}")
        print(f"Design matrix rank: {np.linalg.matrix_rank(design_matrix)}")
        print(f"Is matrix singular? {np.linalg.matrix_rank(design_matrix) < design_matrix.shape[1]}")
        
        # Check for empty cells
        cell_counts = np.sum(design_matrix, axis=0)
        print(f"Cell counts: {cell_counts}")
        
        # Check if any cells are empty
        empty_cells = np.where(cell_counts == 0)[0]
        if len(empty_cells) > 0:
            print(f"WARNING: Empty cells found at indices: {empty_cells}")
            print("This will cause the matrix to be singular!")
        else:
            print("✓ All cells have subjects - matrix should be non-singular")
        
        return design_matrix, factor_levels
    
    return None, None

def main():
    """Main function to test the design matrix generation."""
    print("=== Testing Placebo Design Matrix Generation (Excluding Trans) ===\n")
    
    # Load data
    df = load_behavioral_data()
    
    # Filter for placebo subjects
    df_placebo = filter_placebo_subjects(df)
    
    # Test with different column combinations
    test_columns = [
        ['subID', 'group_id', 'gender_id'],
        ['subID', 'group_id'],
        ['subID', 'gender_id']
    ]
    
    for columns in test_columns:
        print(f"\n{'='*50}")
        print(f"Testing columns: {columns}")
        print(f"{'='*50}")
        
        try:
            design_matrix, factor_levels = create_design_matrix(df_placebo, columns)
            if design_matrix is not None:
                print(f"✓ Design matrix created successfully")
            else:
                print(f"✗ Design matrix creation failed")
        except Exception as e:
            print(f"✗ Error creating design matrix: {e}")
        
        print()

if __name__ == "__main__":
    main()
