import numpy as np
import matplotlib.pyplot as plt
from features import team_tint_color
from data_loader import load_and_clean_data

df = load_and_clean_data()

# ---------- helpers ----------
def _pick_snapshot_frame(play_df, event_preference=("handoff", "ball_snap"), frame_offset_after_handoff=2):
    if 'FrameId' not in play_df.columns or 'event' not in play_df.columns:
        return None
    for ev in event_preference:
        m = (play_df['event'] == ev)
        if m.any():
            base = int(play_df.loc[m, 'FrameId'].iloc[0])
            if ev == "handoff":
                cand = base + int(frame_offset_after_handoff)
                if (play_df['FrameId'] == cand).any():
                    return cand
            return base
    return int(play_df['FrameId'].min())

def _infer_ball_side(play_df):
    poss = play_df['PossessionTeam'].iloc[0] if 'PossessionTeam' in play_df.columns else None
    home = play_df['HomeTeamAbbr'].iloc[0] if 'HomeTeamAbbr' in play_df.columns else None
    away = play_df['VisitorTeamAbbr'].iloc[0] if 'VisitorTeamAbbr' in play_df.columns else None
    if poss is None:
        return None
    if home is not None and poss == home:
        return 'home'
    if away is not None and poss == away:
        return 'away'
    return None

def _get_first_down_info(play_df, ball_side):
    if ball_side in ("home", "away") and "Team" in play_df.columns:
        offense = play_df[play_df["Team"] == ball_side]
        if offense.empty:
            offense = play_df
    else:
        offense = play_df
    los_x = float(offense["X"].median()) if 'X' in offense.columns else 0.0
    togo = None
    for c in ("YardsToGo", "ToGo", "Distance"):
        if c in play_df.columns:
            try:
                togo = float(play_df[c].iloc[0])
                break
            except Exception:
                pass
    if togo is None:
        return los_x, None
    sign = 1.0
    if "PlayDirection" in play_df.columns:
        direction = str(play_df["PlayDirection"].iloc[0]).lower()
        sign = 1.0 if direction == "right" else -1.0
    return los_x, (los_x + sign * togo)

# ---------- main ----------
def plot_player_coordinates(
    play_id,
    show_vectors=True,
    vec_scale=40.0,      # arrow length multiplier (increase to extend stems)
    speed_cap=9.41,      # observed maximum speed in dataset (yds/s)
    markersize=6,        # small dots
    edge_offset=0.7,     # yards to start arrow away from dot center (≈ dot radius)
    frame_offset_after_handoff=2,
    event_preference=("handoff", "ball_snap")
):
    """
    Arrows start just outside each dot and point outward.
    """
    play_df = df[df['PlayId'] == play_id].copy()
    if play_df.empty:
        raise ValueError(f'No rows found for PlayId={play_id}.')

    snap_frame = _pick_snapshot_frame(play_df, event_preference, frame_offset_after_handoff)
    if snap_frame is not None:
        play_df = play_df[play_df['FrameId'] == snap_frame].copy()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(-10, 120)
    ax.set_ylim(-10, 70)
    ax.set_aspect('equal')

    ball_side = _infer_ball_side(play_df)

    # LOS / 1st down
    try:
        los_x, first_down_x = _get_first_down_info(play_df, ball_side)
        ax.axvline(los_x, color='white', linewidth=2, label='LOS', zorder=1, alpha=0.9)
        if first_down_x is not None:
            ax.axvline(max(0, min(120, first_down_x)), color='orange', linestyle='--', linewidth=2, label='1st Down', zorder=1)
    except Exception:
        pass

    rusher_id = play_df['NflIdRusher'].iloc[0] if 'NflIdRusher' in play_df.columns else None
    has_nflid = 'NflId' in play_df.columns

    # we'll draw arrows directly per-player so lengths reflect computed L
    n_with_dir = 0

    for _, row in play_df.iterrows():
        x = float(row['X']) if 'X' in row and not np.isnan(row['X']) else 0.0
        y = float(row['Y']) if 'Y' in row and not np.isnan(row['Y']) else 0.0
        s = float(row['S']) if 'S' in row and not np.isnan(row['S']) else 0.0
        team = row['Team'] if 'Team' in row else None

        base = plt.cm.viridis(min(s, speed_cap) / max(speed_cap, 1e-6))
        team_type = ('home' if (ball_side is not None and team == ball_side)
                     else ('away' if ball_side is not None else (team if team in ('home', 'away') else 'home')))

        if rusher_id is not None and has_nflid and row['NflId'] == rusher_id:
            color = (1.0, 0.84, 0.0)
            z = 4
        else:
            color = team_tint_color(base, team_type)
            z = 3

        ax.plot(x, y, 'o', color=color, markersize=markersize, zorder=z)

        if show_vectors and ('Dir' in row) and (not np.isnan(row['Dir'])):
            n_with_dir += 1
            s_clamped = min(s, speed_cap)
            rad = np.deg2rad(float(row['Dir']))
            dx = np.cos(rad)
            dy = -np.sin(rad)  # unit direction in plot coords
            mag = np.hypot(dx, dy)
            if mag < 1e-6:
                continue
            ux, uy = dx / mag, dy / mag

            # Linear mapping: arrow length proportional to speed
            # L will be vec_scale when s_clamped == speed_cap
            L = vec_scale * (s_clamped / max(speed_cap, 1e-6))
            # Minimum visible length
            L = max(L, 3.0)
            x0 = x + ux * edge_offset
            y0 = y + uy * edge_offset

            dx = ux * L
            dy = uy * L
            # draw arrow: thin shaft (width), small head (head_width/head_length)
            shaft_width = 0.06  # in data units (yards) - thin line
            head_w = 0.6
            head_l = 1.2
            ax.arrow(
                x0, y0, dx, dy,
                length_includes_head=True,
                linewidth=0.8,
                width=shaft_width,
                head_width=head_w,
                head_length=head_l,
                fc=color, ec=color,
                alpha=0.95,
                zorder=9
            )
        # (speed labels removed) 

    # no quiver aggregation; arrows drawn inline above
        # ax.quiverkey(Q, X=0.88, Y=1.03, U=vec_scale*5, label='≈5 yds/s', labelpos='E')

    ttl = f'Player Positions for PlayId: {play_id}'
    if snap_frame is not None:
        ttl += f' | Frame {snap_frame}'
    ax.set_title(ttl)
    ax.set_xlabel('X Coordinate (yards)')
    ax.set_ylabel('Y Coordinate (yards)')
    ax.legend(loc='upper right', framealpha=0.85)
    ax.grid(True, alpha=0.3)
    plt.show()

    if show_vectors and n_with_dir == 0:
        print("Note: No valid 'Dir' values found for this snapshot. Try another frame or check your data.")

# ---- example ----
if __name__ == "__main__":
    plot_player_coordinates(
        play_id=20170907000118,
        show_vectors=True,
        vec_scale=40.0,
        markersize=6,
        edge_offset=0.7
    )