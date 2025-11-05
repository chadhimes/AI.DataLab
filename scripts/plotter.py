import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from features import team_tint_color
from data_loader import load_and_clean_data

# Load data once (can be reloaded in __main__ if desired)
df = load_and_clean_data()

# If you want to hard-code a PlayId inside this script (instead of passing via
# the terminal), set DEFAULT_PLAY_ID to that value (int or string). When not
# None the script will use this PlayId and skip prompting.
DEFAULT_PLAY_ID = 20171207002673


def _playid_to_str(val):
    """Normalize PlayId to a comparable string form.

    Handles ints, large floats and string values. If float is integral, cast to int
    to avoid trailing .0 representation differences from CSV parsing.
    """
    try:
        if val is None:
            return ""
        if isinstance(val, float):
            if np.isnan(val):
                return ""
            if float(val).is_integer():
                return str(int(val))
            # Non-integer float: use full repr
            return repr(val)
        if isinstance(val, (int, np.integer)):
            return str(int(val))
        return str(val)
    except Exception:
        return str(val)


def _get_play_mask(play_id, df_ref):
    """Return a boolean mask for rows in df_ref matching play_id.

    This tries several safe matching strategies:
      1. Exact string match via _playid_to_str
      2. Integer match (cast both sides to int when possible)
      3. If play_id looks like an index (small integer) and no match found,
         select the nth unique PlayId (preserves existing behavior).
    Prints debug information about which strategy matched.
    """
    try:
        play_id_str = _playid_to_str(play_id)
        # 1) exact string match
        s = df_ref['PlayId'].apply(_playid_to_str)
        mask = s == play_id_str
        if mask.any():
            print(f"[debug-match] used exact string match for PlayId '{play_id_str}' -> {mask.sum()} rows")
            return mask

        # 2) try integer match if play_id looks numeric
        try:
            play_id_int = int(float(play_id))
            def _to_int_maybe(v):
                try:
                    if pd.isnull(v):
                        return None
                    return int(float(v))
                except Exception:
                    return None
            ints = df_ref['PlayId'].apply(_to_int_maybe)
            mask2 = ints == play_id_int
            if mask2.any():
                print(f"[debug-match] used integer match for PlayId {play_id_int} -> {mask2.sum()} rows")
                return mask2.fillna(False)
        except Exception:
            pass

        # 3) allow index-based selection when play_id is a small integer index
        if isinstance(play_id, (int, np.integer)):
            unique_ids = df_ref['PlayId'].unique()
            if 0 <= play_id < len(unique_ids):
                selected = unique_ids[play_id]
                print(f"[debug-match] using PlayId index {play_id} -> {selected}")
                sel_str = _playid_to_str(selected)
                return df_ref['PlayId'].apply(_playid_to_str) == sel_str

        # No match
        print(f"[debug-match] no match found for PlayId '{play_id_str}' (tried string/int/index)")
        return pd.Series(False, index=df_ref.index)
    except Exception as e:
        print(f"[debug-match] error while matching PlayId: {e}")
        return pd.Series(False, index=df_ref.index)


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


def _normalize_play_direction(play_df, target='right'):
    """Return a copy of play_df where the offense is moving toward `target` ('right' or 'left').

    If the play's PlayDirection is opposite the target, flip X across the 120-yard field
    and adjust Dir angles so vectors remain correct.
    """
    df2 = play_df.copy()
    try:
        if 'PlayDirection' not in df2.columns:
            return df2
        current = str(df2['PlayDirection'].iloc[0]).lower()
        if current == target:
            return df2

        # flip horizontally across 120-yard field
        if 'X' in df2.columns:
            df2['X'] = 120.0 - df2['X']

        # Horizontal flip: Dir -> (180 - Dir) % 360
        if 'Dir' in df2.columns:
            def _flip_dir(d):
                try:
                    if pd.isnull(d):
                        return d
                    return (180.0 - float(d)) % 360.0
                except Exception:
                    return np.nan
            df2['Dir'] = df2['Dir'].apply(_flip_dir)

        df2['PlayDirection'] = target
    except Exception:
        return play_df.copy()
    return df2

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
    # Robust PlayId matching using helper that attempts string/int/index strategies
    mask = _get_play_mask(play_id, df)
    play_df = df[mask].copy()
    matched_playids = play_df['PlayId'].unique().tolist() if not play_df.empty else []
    print(f"[debug] Requested PlayId: {play_id!r}, matched PlayIds: {matched_playids}, rows: {len(play_df)}")
    if play_df.empty:
        sample_ids = df['PlayId'].head(20).tolist()
        print(f"Requested PlayId: {play_id!r} not found. First 20 PlayIds:")
        print(sample_ids)
        raise ValueError(f'No rows found for PlayId={play_id}.')

    snap_frame = _pick_snapshot_frame(play_df, event_preference, frame_offset_after_handoff)
    if snap_frame is not None:
        play_df = play_df[play_df['FrameId'] == snap_frame].copy()

    # Normalize play direction so offense always moves to the right
    play_df = _normalize_play_direction(play_df, target='right')

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(-10, 120)
    ax.set_ylim(-10, 70)
    ax.set_aspect('equal')

    ball_side = _infer_ball_side(play_df)

    # LOS / 1st down (blue dashed LOS, yellow dashed 1st down)
    try:
        los_x, first_down_x = _get_first_down_info(play_df, ball_side)
        ax.axvline(los_x, color='blue', linestyle='--', linewidth=2, label='LOS', zorder=1, alpha=0.9)
        if first_down_x is not None:
            ax.axvline(max(0, min(120, first_down_x)), color='yellow', linestyle='--', linewidth=2, label='1st Down', zorder=1)
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
            L = max(L, 2.5)
            # Tail at player center
            x0 = x
            y0 = y

            dx = ux * L
            dy = uy * L
            # thinner shaft and smaller head for legibility
            shaft_width = 0.03
            head_w = 0.45
            head_l = 0.9
            ax.arrow(
                x0, y0, dx, dy,
                length_includes_head=True,
                linewidth=0.6,
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
    # Build a small legend/info box with environmental and game context
    info_keys = [
        'GameWeather', 'Weather', 'Temperature', 'WindSpeed', 'WindDirection', 'Humidity',
        'Stadium', 'StadiumType', 'Surface'
    ]

    info_lines = []
    for k in info_keys:
        if k in play_df.columns:
            val = play_df[k].dropna().unique()
            if len(val) > 0:
                info_lines.append(f"{k}: {val[0]}")
    if info_lines:
        info_text = '\n'.join(info_lines)
        ax.text(1.01, 0.5, info_text, transform=ax.transAxes, fontsize=9,
                va='center', ha='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    ax.set_xlabel('X Coordinate (yards)')
    ax.set_ylabel('Y Coordinate (yards)')
    ax.legend(loc='upper right', framealpha=0.85)
    ax.grid(True, alpha=0.3)
    plt.show()

    if show_vectors and n_with_dir == 0:
        print("Note: No valid 'Dir' values found for this snapshot. Try another frame or check your data.")

# ---- example / quick runner ----
if __name__ == "__main__":
    # Reload data and clear previous figures to ensure fresh run
    import argparse
    import os
    plt.close('all')
    df = load_and_clean_data()

    # Diagnostic: print which file is running and the runtime DEFAULT_PLAY_ID
    try:
        print(f"[diag] running file: {os.path.abspath(__file__)}")
        print(f"[diag] DEFAULT_PLAY_ID at runtime: {DEFAULT_PLAY_ID!r}")
        print(f"[diag] file mtime: {os.path.getmtime(os.path.abspath(__file__))}")
    except Exception:
        pass

    # Only show the PlayId list when DEFAULT_PLAY_ID isn't set
    if DEFAULT_PLAY_ID is None:
        unique_playids = df['PlayId'].unique()
        print("First 20 PlayIds (copy one to set below):")
        print(unique_playids[:20])
    else:
        unique_playids = df['PlayId'].unique()

    # If DEFAULT_PLAY_ID is set at the top of this file, use it and skip prompting
    try:
        if DEFAULT_PLAY_ID is not None:
            print(f"[info] Using DEFAULT_PLAY_ID from script: {DEFAULT_PLAY_ID!r}")
            plot_player_coordinates(
                play_id=DEFAULT_PLAY_ID,
                show_vectors=True,
                vec_scale=40.0,
                markersize=6,
                edge_offset=0.7
            )
            raise SystemExit(0)
    except NameError:
        # DEFAULT_PLAY_ID not defined for some reason; fall back to interactive mode
        pass

    parser = argparse.ArgumentParser(description='Plot a single play from train.csv')
    parser.add_argument('--play_id', '-p', default=None,
                        help="PlayId to plot. Can be a PlayId value, a numeric string, or an index into the unique PlayId list (prefix with 'i:' e.g. 'i:3').")
    args = parser.parse_args()

    raw = args.play_id
    if raw is None:
        raw = input("Enter PlayId (or i:<index>) (empty to use first shown): ").strip()
        if raw == "":
            raw = unique_playids[0]

    # Interpret input
    play_id = raw
    try:
        if isinstance(raw, str) and raw.startswith(('i:', 'idx:')):
            idx = int(raw.split(':', 1)[1])
            play_id = unique_playids[idx]
        else:
            # Prefer interpreting obvious numeric strings as integers
            if isinstance(raw, str):
                try:
                    if raw.isdigit():
                        play_id = int(raw)
                    else:
                        # handle scientific or decimal forms
                        play_id = int(float(raw))
                except Exception:
                    play_id = raw
    except Exception as e:
        print(f"Could not parse requested play_id '{raw}': {e}. Falling back to raw input.")
        play_id = raw

    print(f"[info] Plotting PlayId: {play_id!r}")
    plot_player_coordinates(
        play_id=play_id,
        show_vectors=True,
        vec_scale=40.0,
        markersize=6,
        edge_offset=0.7
    )