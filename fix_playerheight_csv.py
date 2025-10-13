#!/usr/bin/env python3
"""
Script to fix the PlayerHeight column in the CSV file.
Converts month-based encoding to proper feet-inches format.
"""

import pandas as pd
import numpy as np
from data_loader import height_to_inches

def fix_playerheight_in_csv(input_file="train.csv", output_file="train_fixed.csv"):
    """
    Fix the PlayerHeight column in the CSV file.
    
    Parameters:
    - input_file: Path to the original CSV file
    - output_file: Path to save the fixed CSV file
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Sample original PlayerHeight values:")
    print(df['PlayerHeight'].value_counts().head(10))
    
    # Convert heights using our fixed function
    print("\nConverting PlayerHeight values...")
    df['PlayerHeightInches'] = df['PlayerHeight'].apply(height_to_inches)
    
    # Create proper feet-inches format for PlayerHeight column
    def inches_to_feet_inches_format(inches):
        """Convert inches back to 'feet-inches' string format."""
        if pd.isnull(inches):
            return np.nan
        feet = int(inches // 12)
        remaining_inches = int(inches % 12)
        return f"{feet}-{remaining_inches:02d}"
    
    # Replace the PlayerHeight column with proper format
    df['PlayerHeight'] = df['PlayerHeightInches'].apply(inches_to_feet_inches_format)
    
    # Remove the temporary PlayerHeightInches column since we fixed the original
    df = df.drop('PlayerHeightInches', axis=1)
    
    print(f"\nFixed PlayerHeight values:")
    print(df['PlayerHeight'].value_counts().head(10))
    
    # Verify the conversion worked
    print(f"\nVerification - checking a few conversions:")
    sample_heights = df['PlayerHeight'].dropna().unique()[:5]
    for height in sample_heights:
        inches = height_to_inches(height)
        if pd.notna(inches):
            feet = int(inches // 12)
            inch_part = int(inches % 12)
            print(f"{height} = {feet} ft {inch_part} in = {inches} inches")
        else:
            print(f"{height} = could not convert (NaN)")
    
    # Save the fixed dataset
    print(f"\nSaving fixed dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print(f"Successfully saved fixed dataset!")
    print(f"Dataset shape: {df.shape}")
    
    # Show file sizes
    import os
    original_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
    new_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"Original file size: {original_size:.2f} MB")
    print(f"Fixed file size: {new_size:.2f} MB")
    
    return df

if __name__ == "__main__":
    # Fix the CSV file
    fixed_df = fix_playerheight_in_csv()
    
    print("\nDone! The PlayerHeight column now contains proper feet-inches format.")
    print("Example: '6-03' means 6 feet 3 inches")
    print("\nTo use the fixed file, update your data_loader.py to load 'train_fixed.csv'")