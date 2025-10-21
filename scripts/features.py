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
    if team_type == "home":
        # Strong blue tint
        tint = np.array([0.1, 0.3, 1.0])
    elif team_type == "away":
        # Strong red tint
        tint = np.array([1.0, 0.1, 0.1])
    else:
        # fallback: no tint
        tint = base_color

    # Blend: 50% base color, 50% tint for more obvious effect
    blended = 0.5 * base_color + 0.5 * tint
    return np.clip(blended, 0, 1)