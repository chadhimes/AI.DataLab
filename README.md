# AI.DataLab - NFL Tracking Data Analysis

## 📁 Project Structure

```
AI.DataLab/
├── 📂 data/           # Dataset files
│   └── train.csv      # NFL tracking data (corrected)
├── 📂 scripts/        # Python scripts and utilities
│   ├── data_loader.py # Main data loading and cleaning functions
│   └── fix_playerheight_csv.py # Script used to fix PlayerHeight column
├── 📂 docs/           # Documentation and reports
│   ├── PlayerHeight_Fix_Summary.md # Details of height column fix
│   └── System_Verification_Report.md # System status verification
└── README.md          # This file
```

## 🚀 Quick Start

### Loading Data
```python
# From project root directory
from scripts.data_loader import load_and_clean_data

df = load_and_clean_data()
print(f"Loaded {len(df):,} rows of NFL tracking data")
```

### From scripts directory
```python
# If running from scripts/ folder
from data_loader import load_and_clean_data

df = load_and_clean_data()
```

## 📊 Dataset Information

- **File**: `data/train.csv`
- **Size**: ~247 MB
- **Rows**: 682,154
- **Columns**: 52
- **Plays**: 31,007 unique plays
- **Players**: 2,570 unique players

## 🔧 Key Features

### Data Loading (`scripts/data_loader.py`)
- Loads and cleans NFL tracking data
- Converts PlayerHeight to inches (fixed format)
- Handles datetime conversions and timezone issues
- Converts movement data to proper formats

### Height Conversion
- PlayerHeight format: "6-03" (6 feet 3 inches)
- Automatically converts to PlayerHeightInches
- Range: 66-81 inches (5'6" - 6'9")

## 📈 Data Quality

✅ **All systems operational**
- PlayIds properly formatted (14-digit integers)
- PlayerHeight corrected and standardized
- 22 players per play (consistent structure)
- Minimal NaN values in critical fields

## 🏈 NFL Data Fields

Key columns include:
- **GameId, PlayId**: Game and play identifiers
- **X, Y**: Player coordinates on field
- **S, A**: Speed and acceleration
- **PlayerHeight, PlayerHeightInches**: Player height data
- **Position**: Player position (QB, RB, WR, etc.)
- **Dir, Orientation**: Movement direction data

## 📝 Documentation

See `docs/` folder for detailed documentation:
- **PlayerHeight_Fix_Summary.md**: How we fixed the corrupted height data
- **System_Verification_Report.md**: Current system status and verification

## 🔄 Usage Examples

### Basic Data Loading
```python
from scripts.data_loader import load_and_clean_data

# Load with default settings
df = load_and_clean_data()

# Check the data
print(df.head())
print(f"Dataset shape: {df.shape}")
print(f"Height range: {df['PlayerHeightInches'].min()}-{df['PlayerHeightInches'].max()} inches")
```

### Working with Play Data
```python
# Get data for a specific play
play_data = df[df['PlayId'] == df['PlayId'].iloc[0]]
print(f"Players in play: {len(play_data)}")

# Analyze height distribution
height_stats = df['PlayerHeightInches'].describe()
print(height_stats)
```

---

*Last updated: October 13, 2025*