import matplotlib.pyplot as plt
import numpy as np
from features import team_tint_color
from data_loader import load_and_clean_data

df = load_and_clean_data()

def plot_player_coordinates(play_id):
    play_df = df[df['PlayId'] == play_id]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-10, 110)
    ax.set_ylim(-10, 110)
    ax.set_aspect('equal')
    for _, row in play_df.iterrows():
        x, y = row['X'], row['Y']
        speed = row['S']
        team = row['Team']
        base_color = plt.cm.viridis(speed / 25)  # Normalize speed for colormap
        color = team_tint_color(base_color, team)
        ax.plot(x, y, 'o', color=color, markersize=10)
    ax.set_title(f'Player Positions for PlayId: {play_id}')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()
    
# Example usage
plot_player_coordinates(play_id=20170907000118)  # Replace with a valid PlayId from your dataset