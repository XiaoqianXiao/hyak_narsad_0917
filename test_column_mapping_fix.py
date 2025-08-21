#!/usr/bin/env python3
"""
Test script to verify the column mapping fix works correctly.
"""

import pandas as pd

def test_column_mapping_fix():
    """Test the fixed column mapping logic."""
    
    print("=== Testing Column Mapping Fix ===\n")
    
    # Simulate the exact scenario from the error log
    print("Step 1: Create test data (simulating the DataFrame after Trans subject exclusion)")
    test_data = [
        ('N101', 1, 1),  # Patient, Female
        ('N102', 1, 2),  # Patient, Male  
        ('N104', 2, 1),  # Control, Female
        ('N105', 2, 2),  # Control, Male
    ]
    
    # This DataFrame has the actual column names from the data
    df_behav = pd.DataFrame(test_data, columns=['subID', 'group_id', 'gender_code'])
    print("DataFrame with actual column names:")
    print(df_behav)
    print(f"Columns: {list(df_behav.columns)}")
    print()
    
    # Step 2: Simulate the column mapping logic
    print("Step 2: Column mapping logic")
    
    # User requests these columns
    final_include_columns = ['subID', 'group_id', 'gender_id']
    print(f"User requested columns: {final_include_columns}")
    
    # Column mapping (same as in the code)
    column_mapping = {
        'gender_id': 'gender_code',  # Map gender_id to gender_code
        'drug_id': 'drug_id',        # Keep drug_id as is
        'group_id': 'group_id',      # Keep group_id as is
        'subID': 'subID'             # Keep subID as is
    }
    
    print(f"Column mapping: {column_mapping}")
    print()
    
    # Step 3: Apply the fixed mapping logic
    print("Step 3: Apply fixed mapping logic")
    
    # Map output column names back to data column names for processing
    processing_columns = []
    for col in final_include_columns:
        if col in column_mapping:
            # Use the actual column name in the data
            processing_columns.append(column_mapping[col])
            print(f"✓ Mapped '{col}' -> '{column_mapping[col]}'")
        else:
            # Keep the column name as is
            processing_columns.append(col)
            print(f"✓ Kept '{col}' as is")
    
    print(f"\nFinal processing columns: {processing_columns}")
    print()
    
    # Step 4: Verify the columns exist in the DataFrame
    print("Step 4: Verify columns exist in DataFrame")
    missing_columns = [col for col in processing_columns if col not in df_behav.columns]
    if missing_columns:
        print(f"❌ Missing columns: {missing_columns}")
        print(f"Available columns: {list(df_behav.columns)}")
        return False
    else:
        print("✓ All processing columns exist in DataFrame")
    
    print()
    
    # Step 5: Test the actual data access
    print("Step 5: Test data access")
    try:
        selected_data = df_behav[processing_columns]
        print("✓ Successfully selected data:")
        print(selected_data)
        print()
        
        # Convert to list of tuples (as in the actual code)
        group_info = list(selected_data.itertuples(index=False, name=None))
        print("✓ Successfully converted to list of tuples:")
        for i, item in enumerate(group_info):
            print(f"  {i+1}: {item}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data access failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing the column mapping fix...\n")
    
    success = test_column_mapping_fix()
    
    print("\n" + "="*50)
    if success:
        print("✅ COLUMN MAPPING FIX TEST PASSED!")
        print("The fix should resolve the KeyError in the actual code.")
    else:
        print("❌ COLUMN MAPPING FIX TEST FAILED!")
        print("The fix needs further adjustment.")
    print("="*50)
