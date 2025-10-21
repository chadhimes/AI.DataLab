# features.py
import matplotlib.pyplot as plt
import numpy as np

def team_tint_color(base_color, team_type):
    """
    Tint the player's speed-based color depending on team:
    - 'offense' → reddish tint
    - 'defense' → bluish tint
    """
    base_color = np.array(base_color[:3])  # RGB only
    if team_type == "offense":
        tint = np.array([1.0, 0.4, 0.4])  # red tint
    else:
        tint = np.array([0.4, 0.6, 1.0])  # blue tint
    
    # Blend: 70% speed color, 30% tint
    blended = 0.7 * base_color + 0.3 * tint
    return np.clip(blended, 0, 1)