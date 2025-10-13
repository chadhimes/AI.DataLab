#!/usr/bin/env python3
"""
Example usage of the NFL data loader
Run from project root: python3 scripts/example_usage.py
"""

from data_loader import load_and_clean_data
import pandas as pd

def main():
    print("🏈 NFL Data Analysis Example")
    print("=" * 40)
    
    # Load the data
    print("Loading NFL tracking data...")
    df = load_and_clean_data()
    
    print(f"✅ Loaded {len(df):,} rows of data")
    print(f"📊 Dataset shape: {df.shape}")
    
    # Basic analysis
    print("\n📈 Basic Statistics:")
    print(f"   • Unique games: {df['GameId'].nunique()}")
    print(f"   • Unique plays: {df['PlayId'].nunique()}")  
    print(f"   • Unique players: {df['NflId'].nunique()}")
    
    # Height analysis
    print("\n📏 Player Height Analysis:")
    height_stats = df['PlayerHeightInches'].describe()
    print(f"   • Average: {height_stats['mean']:.1f} inches ({height_stats['mean']//12:.0f}' {height_stats['mean']%12:.0f}\")")
    print(f"   • Range: {height_stats['min']:.0f}-{height_stats['max']:.0f} inches")
    
    # Position analysis
    print("\n🏈 Player Positions:")
    position_counts = df['Position'].value_counts().head(10)
    for pos, count in position_counts.items():
        print(f"   • {pos}: {count:,} records")
    
    # Sample play
    print("\n🎯 Sample Play Data:")
    sample_play = df[df['PlayId'] == df['PlayId'].iloc[0]]
    print(f"   • PlayId: {sample_play['PlayId'].iloc[0]}")
    print(f"   • Players: {len(sample_play)}")
    print(f"   • Positions: {', '.join(sample_play['Position'].unique())}")
    
    print("\n✅ Analysis complete!")
    
    return df

if __name__ == "__main__":
    df = main()