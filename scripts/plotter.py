import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from features import team_tint_color
from data_loader import load_and_clean_data

# NEW: symbol panel imports
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
import math
import re
import os
import sys
import random

# Load data once (can be reloaded in __main__ if desired)
df = load_and_clean_data()

# Optional default (set to None to use CLI / interactive)
DEFAULT_PLAY_ID = 20181209063193


# ---------- PlayId robust matching ----------
def _playid_to_str(val):
    try:
        if val is None:
            return ""
        if isinstance(val, float):
            if np.isnan(val):
                return ""
            if float(val).is_integer():
                return str(int(val))
            return repr(val)
        if isinstance(val, (int, np.integer)):
            return str(int(val))
        return str(val)
    except Exception:
        return str(val)

def _get_play_mask(play_id, df_ref):
    try:
        play_id_str = _playid_to_str(play_id)
        s = df_ref['PlayId'].apply(_playid_to_str)
        mask = s == play_id_str
        if mask.any():
            print(f"[debug-match] exact string match '{play_id_str}' -> {mask.sum()} rows")
            return mask
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
                print(f"[debug-match] integer match {play_id_int} -> {mask2.sum()} rows")
                return mask2.fillna(False)
        except Exception:
            pass
        if isinstance(play_id, (int, np.integer)):
            unique_ids = df_ref['PlayId'].unique()
            if 0 <= play_id < len(unique_ids):
                selected = unique_ids[play_id]
                print(f"[debug-match] index {play_id} -> {selected}")
                sel_str = _playid_to_str(selected)
                return df_ref['PlayId'].apply(_playid_to_str) == sel_str
        print(f"[debug-match] no match for '{play_id_str}'")
        return pd.Series(False, index=df_ref.index)
    except Exception as e:
        print(f"[debug-match] error matching PlayId: {e}")
        return pd.Series(False, index=df_ref.index)


# ---------- snapshot helpers ----------
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
                togo = float(play_df[c].iloc[0]); break
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
    """Mirror X and rotate Dir if needed so offense moves toward `target`."""
    df2 = play_df.copy()
    try:
        if 'PlayDirection' not in df2.columns:
            return df2
        current = str(df2['PlayDirection'].iloc[0]).lower()
        if current == target:
            return df2
        if 'X' in df2.columns:
            df2['X'] = 120.0 - df2['X']
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


# ======= SYMBOL-ONLY ENV PANEL (no text) =======
def _extract_first_nonnull(play_df, keys):
    for k in keys:
        if k in play_df.columns:
            vals = play_df[k].dropna().astype(str)
            if len(vals):
                return vals.iloc[0]
    return None

def _parse_weather_category(play_df):
    raw = _extract_first_nonnull(play_df, ['GameWeather', 'Weather'])
    if not raw:
        return 'cloudy'
    s = raw.lower()
    if any(w in s for w in ['rain', 'showers', 'drizzle', 'storm', 'wet']):
        return 'rainy'
    if any(w in s for w in ['sun', 'clear', 'fair']):
        return 'sunny'
    return 'cloudy'

def _parse_wind(play_df):
    raw_dir = _extract_first_nonnull(play_df, ['WindDirection'])
    raw_spd = _extract_first_nonnull(play_df, ['WindSpeed'])
    spd = 0.0
    if raw_spd:
        m = re.search(r'(-?\d+(\.\d+)?)', raw_spd)
        if m:
            spd = float(m.group(1))
    deg = 0.0
    if raw_dir:
        m = re.search(r'(-?\d+(\.\d+)?)', raw_dir)
        if m:
            deg = float(m.group(1))
        else:
            card = raw_dir.strip().upper()
            compass = {
                'N':0,'NNE':22.5,'NE':45,'ENE':67.5,'E':90,'ESE':112.5,'SE':135,'SSE':157.5,
                'S':180,'SSW':202.5,'SW':225,'WSW':247.5,'W':270,'WNW':292.5,'NW':315,'NNW':337.5
            }
            deg = compass.get(card, 0.0)
    return deg % 360.0, max(0.0, spd)

def _parse_surface(play_df):
    raw = _extract_first_nonnull(play_df, ['Surface'])
    if not raw:
        return 'natural'
    s = raw.lower()
    if any(w in s for w in ['art', 'turf']):
        return 'artificial'
    return 'natural'

def _parse_stadium_type(play_df):
    raw = _extract_first_nonnull(play_df, ['StadiumType'])
    if not raw:
        return 'outdoor'
    s = raw.lower()
    if any(w in s for w in ['indoor', 'dome', 'closed']):
        return 'indoor'
    if any(w in s for w in ['outdoor', 'open']):
        return 'outdoor'
    return 'outdoor'

def _add_symbol_env_panel(ax, play_df, corner='tr'):
    import matplotlib.patches as patches
    import matplotlib.transforms as mtransforms
    import math
    import re

    # Slightly bigger panel + padding
    W, H = 0.18, 0.14          # was 0.16, 0.12
    PAD = 0.012
    if corner == 'tl':
        x0, y0 = PAD, 1 - PAD - H
    elif corner == 'tr':
        x0, y0 = 1 - PAD - W, 1 - PAD - H
    elif corner == 'bl':
        x0, y0 = PAD, PAD
    else:
        x0, y0 = 1 - PAD - W, PAD

    panel = patches.FancyBboxPatch(
        (x0, y0), W, H,
        transform=ax.transAxes,
        boxstyle="round,pad=0.004,rounding_size=0.006",
        linewidth=0.8, edgecolor="black", facecolor="white",
        zorder=20, clip_on=False    # prevent clipping at axes edge
    )
    ax.add_patch(panel)
    tx = mtransforms.Affine2D().scale(W, H).translate(x0, y0) + ax.transAxes

    # --- helpers to read values already in your file ---
    def _extract_first_nonnull(play_df, keys):
        for k in keys:
            if k in play_df.columns:
                vals = play_df[k].dropna().astype(str)
                if len(vals):
                    return vals.iloc[0]
        return None

    def _parse_weather_category(play_df):
        raw = _extract_first_nonnull(play_df, ['GameWeather', 'Weather'])
        if not raw: return 'cloudy'
        s = raw.lower()
        if any(w in s for w in ['rain', 'showers', 'drizzle', 'storm', 'wet']): return 'rainy'
        if any(w in s for w in ['sun', 'clear', 'fair']): return 'sunny'
        return 'cloudy'

    def _parse_wind(play_df):
        raw_dir = _extract_first_nonnull(play_df, ['WindDirection'])
        raw_spd = _extract_first_nonnull(play_df, ['WindSpeed'])
        spd = 0.0
        if raw_spd:
            m = re.search(r'(-?\d+(\.\d+)?)', raw_spd)
            if m: spd = float(m.group(1))
        deg = 0.0
        if raw_dir:
            m = re.search(r'(-?\d+(\.\d+)?)', raw_dir)
            if m: deg = float(m.group(1))
            else:
                card = raw_dir.strip().upper()
                compass = {'N':0,'NNE':22.5,'NE':45,'ENE':67.5,'E':90,'ESE':112.5,'SE':135,'SSE':157.5,
                           'S':180,'SSW':202.5,'SW':225,'WSW':247.5,'W':270,'WNW':292.5,'NW':315,'NNW':337.5}
                deg = compass.get(card, 0.0)
        return deg % 360.0, max(0.0, spd)

    def _parse_surface(play_df):
        raw = _extract_first_nonnull(play_df, ['Surface'])
        if not raw: return 'natural'
        s = raw.lower()
        return 'artificial' if any(w in s for w in ['art', 'turf']) else 'natural'

    def _parse_stadium_type(play_df):
        raw = _extract_first_nonnull(play_df, ['StadiumType'])
        if not raw: return 'outdoor'
        s = raw.lower()
        if any(w in s for w in ['indoor', 'dome', 'closed']): return 'indoor'
        if any(w in s for w in ['outdoor', 'open']): return 'outdoor'
        return 'outdoor'

    # 1) Weather strip (top-left)
    wx = 0.06; wy = 0.76; ww = 0.46; wh = 0.18
    order = ['sunny', 'cloudy', 'rainy']
    weather = _parse_weather_category(play_df)
    cell_w = ww / 3.0
    ax.add_patch(patches.Rectangle((wx, wy), ww, wh, transform=tx,
                                   fill=False, linewidth=0.6, zorder=21, clip_on=False))
    for i, name in enumerate(order):
        filled = (name == weather)
        ax.add_patch(
            patches.Rectangle((wx + i*cell_w, wy), cell_w, wh,
                              transform=tx, facecolor=('black' if filled else 'white'),
                              edgecolor='black', linewidth=0.6, zorder=22, clip_on=False)
        )

    # 2) Wind compass (top-right)
    cx = 0.80; cy = 0.84; cr = 0.085
    ax.add_patch(patches.Circle((cx, cy), cr, transform=tx,
                                fill=False, linewidth=0.6, zorder=21, clip_on=False))
    dir_deg, spd = _parse_wind(play_df)
    to_deg = (dir_deg + 180.0) % 360.0
    rad = math.radians(to_deg)
    L = cr * (0.5 if spd <= 3 else 0.8 if spd <= 12 else 1.1)
    dx, dy = L * math.cos(rad), L * math.sin(rad)
    ax.add_patch(patches.FancyArrow(cx, cy, dx, dy,
                                    width=0.004, head_width=0.028, head_length=0.038,
                                    length_includes_head=True, transform=tx, color='black',
                                    zorder=22, clip_on=False))

    # 3) Thermometer (lowered so top never clips)
    bx = 0.06
    by = 0   # was 0.36 → move DOWN a bit
    bw = 0.07
    bh = 0.50   # was 0.52 → slightly shorter so top clears the weather strip

    ax.add_patch(patches.Rectangle((bx, by), bw, bh, transform=tx,
                                fill=False, linewidth=0.6, zorder=21, clip_on=False))

    temp_raw = _extract_first_nonnull(play_df, ['Temperature'])
    try:
        temp = float(re.search(r'(-?\d+(\.\d+)?)', str(temp_raw)).group(1)) if temp_raw else 70.0
    except Exception:
        temp = 70.0

    # scale 0–110°F into [0,1] of the thermometer height
    t01 = min(max((temp - 0.0) / 110.0, 0.0), 1.0)
    ax.add_patch(patches.Rectangle((bx, by), bw, bh * t01, transform=tx,
                                facecolor='black', edgecolor='black', linewidth=0.0,
                                zorder=22, clip_on=False))

    bulb_r = bw * 0.45
    ax.add_patch(patches.Circle((bx + bw/2.0, by), radius=bulb_r, transform=tx,
                                color='black', zorder=22, clip_on=False))



    # 4) Surface (bottom center)
    sx = 0.24; sy = 0.12; sw = 0.14; sh = 0.14
    surface = _parse_surface(play_df)
    ax.add_patch(patches.Rectangle((sx, sy), sw, sh, transform=tx,
                                   facecolor=('black' if surface == 'artificial' else 'white'),
                                   edgecolor='black', linewidth=0.6, zorder=21, clip_on=False))

    # 5) Stadium (bottom-right)
    gx = 0.62; gy = 0.12; gw = 0.26; gh = 0.18
    stype = _parse_stadium_type(play_df)
    ax.add_patch(patches.Rectangle((gx, gy), gw, gh, transform=tx,
                                   facecolor='white', edgecolor='black', linewidth=0.6,
                                   zorder=21, clip_on=False))
    if stype == 'indoor':
        ax.add_patch(patches.Polygon([(gx, gy+gh), (gx+gw/2, gy+gh+gh*0.5), (gx+gw, gy+gh)],
                                     closed=True, transform=tx, facecolor='black',
                                     edgecolor='black', zorder=22, clip_on=False))


# ---------- main plot ----------
def plot_player_coordinates(
    play_id,
    show_vectors=True,
    vec_scale=40.0,
    speed_cap=9.41,
    markersize=6,
    edge_offset=0.7,
    frame_offset_after_handoff=2,
    event_preference=("handoff", "ball_snap"),
    train_mode=False,
    save_path=None,
    dpi=200
):
    """
    If train_mode:
      - fixed colors (offense=black, defense=white, rusher=gold), no antialiasing
      - no axes/grid/labels/title (tile-clean)
      - consistent margins (full canvas)
    Otherwise: keep your normal visualization behavior.
    """
    # Match play rows
    mask = _get_play_mask(play_id, df)
    play_df = df[mask].copy()
    if play_df.empty:
        raise ValueError(f'No rows found for PlayId={play_id}.')

    # Pick frame and normalize to offense → right
    snap_frame = _pick_snapshot_frame(play_df, event_preference, frame_offset_after_handoff)
    if snap_frame is not None:
        play_df = play_df[play_df['FrameId'] == snap_frame].copy()
    play_df = _normalize_play_direction(play_df, target='right')

    # Figure
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(-10, 110); ax.set_ylim(-10, 70); ax.set_aspect('equal')

    # Train mode visuals
    if train_mode:
        ax.grid(False)
        ax.set_axis_off()

    # LOS / 1st down
    ball_side = _infer_ball_side(play_df)
    try:
        los_x, first_down_x = _get_first_down_info(play_df, ball_side)
        ax.axvline(los_x, color='black' if train_mode else 'blue',
                   linestyle='--', linewidth=2, zorder=1, alpha=1.0 if train_mode else 0.9)
        if first_down_x is not None:
            ax.axvline(max(0, min(120, first_down_x)),
                       color='black' if train_mode else 'yellow',
                       linestyle='--', linewidth=2, zorder=1, alpha=1.0 if train_mode else 1.0)
    except Exception:
        pass

    rusher_id = play_df['NflIdRusher'].iloc[0] if 'NflIdRusher' in play_df.columns else None
    has_nflid = 'NflId' in play_df.columns

    n_with_dir = 0

    for _, row in play_df.iterrows():
        x = float(row['X']) if 'X' in row and not np.isnan(row['X']) else 0.0
        y = float(row['Y']) if 'Y' in row and not np.isnan(row['Y']) else 0.0
        s = float(row['S']) if 'S' in row and not np.isnan(row['S']) else 0.0
        team = row['Team'] if 'Team' in row else None

        # Colors: train-mode fixed, else original tinting
        if train_mode:
            OFF_DOT = (0, 0, 0)          # offense -> black
            DEF_DOT = (1, 1, 1)          # defense -> white
            RUSHER  = (1.0, 0.84, 0.0)   # highlight
            is_rusher = (rusher_id is not None and has_nflid and row['NflId'] == rusher_id)
            team_type = ('home' if (ball_side is not None and team == ball_side)
                         else ('away' if ball_side is not None else (team if team in ('home','away') else 'home')))
            color = RUSHER if is_rusher else (OFF_DOT if team_type == 'home' else DEF_DOT)
            mec = (0,0,0)
        else:
            base = plt.cm.viridis(min(s, speed_cap) / max(speed_cap, 1e-6))
            team_type = ('home' if (ball_side is not None and team == ball_side)
                         else ('away' if ball_side is not None else (team if team in ('home', 'away') else 'home')))
            if rusher_id is not None and has_nflid and row['NflId'] == rusher_id:
                color = (1.0, 0.84, 0.0)
            else:
                color = team_tint_color(base, team_type)
            mec = color

        ax.plot(x, y, 'o',
                color=color,
                markersize=markersize,
                zorder=3 + (1 if (rusher_id is not None and has_nflid and row.get('NflId', None) == rusher_id) else 0),
                markeredgecolor=(0,0,0) if train_mode else mec,
                markeredgewidth=(0.8 if train_mode else 0.0),
                antialiased=not train_mode)

        # Speed-direction arrows
        if show_vectors and ('Dir' in row) and (not np.isnan(row['Dir'])):
            n_with_dir += 1
            s_clamped = min(s, speed_cap)
            rad = np.deg2rad(float(row['Dir']))
            dx = np.cos(rad); dy = -np.sin(rad)
            mag = np.hypot(dx, dy)
            if mag < 1e-6: continue
            ux, uy = dx / mag, dy / mag
            L = vec_scale * (s_clamped / max(speed_cap, 1e-6))
            L = max(L, 2.5)
            x0, y0 = x, y
            ax.arrow(x0, y0, ux * L, uy * L,
                     length_includes_head=True,
                     linewidth=0.6,
                     width=0.03,
                     head_width=0.45,
                     head_length=0.9,
                     fc=color if not train_mode else (0,0,0),
                     ec=color if not train_mode else (0,0,0),
                     alpha=1.0 if train_mode else 0.95,
                     zorder=9,
                     antialiased=not train_mode)

    # Title / labels for non-train mode
    if not train_mode:
        ttl = f'Player Positions for PlayId: {play_id}'
        if snap_frame is not None:
            ttl += f' | Frame {snap_frame}'
        ax.set_title(ttl)
        ax.set_xlabel('X Coordinate (yards)')
        ax.set_ylabel('Y Coordinate (yards)')
        ax.legend(loc='upper right', framealpha=0.85)
        ax.grid(True, alpha=0.3)

    # Symbol-only env panel always (monochrome, stable)
    _add_symbol_env_panel(ax, play_df, corner='tr')

    # Show or save
    if save_path is None:
        plt.show()
    else:
        fig.subplots_adjust(0,0,1,1)
        plt.savefig(save_path, dpi=dpi, facecolor='white',
                    edgecolor='none', bbox_inches=None, pad_inches=0)
        plt.close(fig)

    if show_vectors and n_with_dir == 0:
        print("Note: No valid 'Dir' values for this snapshot.")


# ---------- CLI / runner ----------
if __name__ == "__main__":
    import argparse
    plt.close('all')

    # Reload fresh
    df = load_and_clean_data()

    parser = argparse.ArgumentParser(description='Plot a single play (CNN-optimized option).')
    parser.add_argument('--play_id', '-p', default=None,
                        help="PlayId value, numeric string, or index 'i:<idx>' into unique PlayIds.")
    parser.add_argument('--train_mode', action='store_true',
                        help="Render CNN-optimized tiles (fixed colors, no axes, no antialiasing).")
    parser.add_argument('--save_dir', type=str, default=None,
                        help="If set, save PNGs to this directory (filename auto-generated).")
    parser.add_argument('--dpi', type=int, default=200, help="DPI when saving.")
    parser.add_argument('--random_n', type=int, default=0,
                        help="If >0, ignore play_id and sample N random unique PlayIds (fresh randomness each run).")
    parser.add_argument('--require_vectors', action='store_true',
                        help="When random_n>0, only choose plays that have any non-NaN 'Dir' somewhere.")
    
    parser.add_argument('--all', action='store_true',
                        help="Export tiles for ALL unique PlayIds.")
    parser.add_argument('--skip_existing', action='store_true',
                        help="Skip rendering if the output file already exists.")
    parser.add_argument('--max_n', type=int, default=None,
                        help="If set with --all, limit to the first N candidates after filtering.")
    parser.add_argument('--progress_every', type=int, default=200,
                        help="Print progress every K renders when using --all.")


    args = parser.parse_args()

        # ----- Auto-trigger "export all" when no args were given -----
    # If user just clicks Run in an IDE (no CLI args), do: export all in train_mode
    if (args.play_id is None) and (args.random_n == 0) and (not args.save_dir) and (not args.all):
        # default output dir: <repo>/out_all_plays
        default_out = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'out_all_plays'))
        args.all = True
        args.train_mode = True          # CNN-friendly tiles by default
        args.save_dir = default_out
        args.skip_existing = True       # don't re-render existing files
        args.progress_every = 500
        print(f"[auto] No args supplied → exporting ALL plays to: {args.save_dir}")


    # Build unique ids
    unique_playids = df['PlayId'].unique()


    # ---- Export ALL plays (optionally filtered) ----
    if args.all:
        if not args.save_dir:
            print("[error] --all requires --save_dir to be set.")
            sys.exit(1)

        out_dir = args.save_dir
        os.makedirs(out_dir, exist_ok=True)

        candidates = df['PlayId'].unique()

        if args.require_vectors:
            def _has_dir(pid):
                sub = df[df['PlayId'] == pid]
                return ('Dir' in sub.columns) and sub['Dir'].notna().any()
            candidates = np.array([pid for pid in candidates if _has_dir(pid)])
            if len(candidates) == 0:
                print("[warn] No plays with any 'Dir' found. Remove --require_vectors.")
                sys.exit(1)

        if args.max_n is not None:
            candidates = candidates[:int(args.max_n)]

        total = len(candidates)
        print(f"[info] Exporting {total} plays to {out_dir} (train_mode={args.train_mode})")

        for i, pid in enumerate(candidates, 1):
            fname = f"play_{_playid_to_str(pid)}_tile.png"
            save_path = os.path.join(out_dir, fname)

            if args.skip_existing and os.path.exists(save_path):
                if (i % args.progress_every) == 0 or i == total:
                    print(f"[{i}/{total}] skipped existing up to here …")
                continue

            try:
                plot_player_coordinates(
                    play_id=pid,
                    show_vectors=True,
                    vec_scale=40.0,
                    markersize=6,
                    edge_offset=0.7,
                    train_mode=args.train_mode,
                    save_path=save_path,
                    dpi=args.dpi
                )
            except Exception as e:
                # Keep going on errors
                print(f"[warn] Failed on PlayId={pid}: {e}")

            if (i % args.progress_every) == 0 or i == total:
                print(f"[{i}/{total}] done")
        sys.exit(0)



    # Random sampling path (no fixed seed → fresh each run)
    if args.random_n and args.random_n > 0:
        candidates = unique_playids
        if args.require_vectors:
            def _has_dir(pid):
                sub = df[df['PlayId'] == pid]
                return ('Dir' in sub.columns) and sub['Dir'].notna().any()
            candidates = np.array([pid for pid in candidates if _has_dir(pid)])
            if len(candidates) == 0:
                print("[warn] No plays with any 'Dir' found. Remove --require_vectors.")
                sys.exit(1)
        k = min(args.random_n, len(candidates))
        sampled = random.sample(list(candidates), k)
        out_dir = args.save_dir
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        for idx, pid in enumerate(sampled, 1):
            print(f"[{idx}/{k}] Plotting {pid} (train_mode={args.train_mode})")
            save_path = None
            if out_dir:
                save_path = os.path.join(out_dir, f"play_{_playid_to_str(pid)}_tile.png")
            plot_player_coordinates(
                play_id=pid,
                show_vectors=True,
                vec_scale=40.0,
                markersize=6,
                edge_offset=0.7,
                train_mode=args.train_mode,
                save_path=save_path,
                dpi=args.dpi
            )
        sys.exit(0)

    # If DEFAULT_PLAY_ID is set and no CLI play_id provided, use it
    play_arg = args.play_id
    if play_arg is None and DEFAULT_PLAY_ID is not None:
        play_arg = DEFAULT_PLAY_ID

    # Interactive fallback
    if play_arg is None:
        print("First 20 PlayIds:")
        print(unique_playids[:20])
        raw = input("Enter PlayId (or i:<index>): ").strip()
    else:
        raw = str(play_arg)

    # Interpret input
    play_id = raw
    try:
        if isinstance(raw, str) and raw.startswith(('i:', 'idx:')):
            idx = int(raw.split(':', 1)[1])
            play_id = unique_playids[idx]
        else:
            if isinstance(raw, str):
                try:
                    if raw.isdigit():
                        play_id = int(raw)
                    else:
                        play_id = int(float(raw))
                except Exception:
                    play_id = raw
    except Exception as e:
        print(f"Could not parse play_id '{raw}': {e}. Using raw input.")
        play_id = raw

    save_path = None
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, f"play_{_playid_to_str(play_id)}_tile.png")

    plot_player_coordinates(
        play_id=play_id,
        show_vectors=True,
        vec_scale=40.0,
        markersize=6,
        edge_offset=0.7,
        train_mode=args.train_mode,
        save_path=save_path,
        dpi=args.dpi
    )
