import matplotlib.pyplot as plt
import numpy as np
from features import team_tint_color
from data_loader import load_and_clean_data

df = load_and_clean_data()

def plot_player_coordinates(play_id):
    play_df = df[df['PlayId'] == play_id]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(-10, 120)
    ax.set_ylim(-10, 70)
    ax.set_aspect('equal')

    # Determine which team has possession of the ball using PossessionTeam (offense in blue, defense in red)
    possession_team = play_df['PossessionTeam'].iloc[0] if 'PossessionTeam' in play_df.columns else None
    visitor_abbr = play_df['VisitorTeamAbbr'].iloc[0] if 'VisitorTeamAbbr' in play_df.columns else None
    home_abbr = play_df['HomeTeamAbbr'].iloc[0] if 'HomeTeamAbbr' in play_df.columns else None

    if possession_team is not None:
        if home_abbr is not None and possession_team == home_abbr:
            ball_side = 'home'
        elif visitor_abbr is not None and possession_team == visitor_abbr:
            ball_side = 'away'
        else:
            ball_side = None
    else:
        ball_side = None

    visitor_abbr = play_df['VisitorTeamAbbr'].iloc[0] if 'VisitorTeamAbbr' in play_df.columns else None
    home_abbr = play_df['HomeTeamAbbr'].iloc[0] if 'HomeTeamAbbr' in play_df.columns else None
    rusher_id = play_df['NflIdRusher'].iloc[0] if 'NflIdRusher' in play_df.columns else None
    for _, row in play_df.iterrows():
        x, y = row['X'], row['Y']
        speed = row['S']
        team = row['Team']
        base_color = plt.cm.viridis(speed / 25)  # Normalize speed for colormap

        # Determine if this player is offense or defense
        if ball_side is not None:
            team_type = 'home' if team == ball_side else 'away'
        else:
            team_type = team  # fallback to original

        # Highlight rusher in gold
        if rusher_id is not None and 'NflId' in row and row['NflId'] == rusher_id:
            color = (1.0, 0.84, 0.0)  # gold
        else:
            color = team_tint_color(base_color, team_type)
        ax.plot(x, y, 'o', color=color, markersize=10)
    ax.set_title(f'Player Positions for PlayId: {play_id}')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    plt.grid(True)
    plt.xlim(0,100)
    plt.ylim(0,60)
    plt.show()

playIDS = df["PlayId"]  
print(playIDS[3])
# Example usage
plot_player_coordinates(play_id=20170907001488)  # Replace with a valid PlayId from your dataset