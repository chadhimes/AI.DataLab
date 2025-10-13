# data_loader.py
import pandas as pd
import numpy as np

def height_to_inches(height_str):
    """Convert height from 'feet-inches' format to total inches (e.g., '6-03' = 75 inches)."""
    if pd.isnull(height_str):
        return np.nan
    
    try:
        height_str = str(height_str).strip()
        
        # Handle standard feet-inches format like "6-03"
        if '-' not in height_str:
            return np.nan
            
        parts = height_str.split('-')
        if len(parts) != 2:
            return np.nan
        
        feet_str, inches_str = parts
        
        # Convert to integers
        feet = int(feet_str)
        inches = int(inches_str)
        
        # Basic sanity check for reasonable height values
        if feet < 1 or feet > 8 or inches < 0 or inches > 11:
            return np.nan
            
        return feet * 12 + inches
        
    except (ValueError, TypeError, AttributeError):
        return np.nan

def load_and_clean_data(file_path=None):
    """Load and clean the dataset."""
    import os
    
    # Set default file path relative to the script location
    if file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "..", "data", "train.csv")
    
    df = pd.read_csv(file_path, low_memory=False)
    
    # Convert height to inches (with error handling for corrupted data)
    df['PlayerHeightInches'] = df['PlayerHeight'].apply(height_to_inches)
    
    # Convert birthdate to age
    df['PlayerBirthDate'] = pd.to_datetime(df['PlayerBirthDate'], errors='coerce')
    df['TimeSnap'] = pd.to_datetime(df['TimeSnap'], errors='coerce')
    
    # Ensure both datetime columns are timezone-naive to avoid comparison issues
    if df['PlayerBirthDate'].dt.tz is not None:
        df['PlayerBirthDate'] = df['PlayerBirthDate'].dt.tz_localize(None)
    if df['TimeSnap'].dt.tz is not None:
        df['TimeSnap'] = df['TimeSnap'].dt.tz_localize(None)
        
    df['PlayerAge'] = (df['TimeSnap'] - df['PlayerBirthDate']).dt.days / 365.25
    
    # Convert movement angles to radians (handle potential missing values)
    df['DirRad'] = np.where(pd.isnull(df['Dir']), np.nan, np.deg2rad(df['Dir']))
    
    # Convert numeric fields
    df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
    df['WindSpeed'] = pd.to_numeric(df['WindSpeed'], errors='coerce')
    df['Humidity'] = pd.to_numeric(df['Humidity'], errors='coerce')
    df['WindDirection'] = pd.to_numeric(df['WindDirection'], errors='coerce')
    
    return df

if __name__ == "__main__":
    df = load_and_clean_data()
    print(df.head())