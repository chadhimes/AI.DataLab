# scripts/make_labels_yards.py
import os, re
import pandas as pd
from data_loader import load_and_clean_data

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
IMG_DIR = os.path.join(ROOT, 'out_all_plays')   # folder with your PNGs

df = load_and_clean_data()

# one row per PlayId
base = df.drop_duplicates(subset=['PlayId']).copy()

# find a yards-gained column that exists in your data
for c in ['YardsGained', 'Yards', 'PlayResult']:
    if c in base.columns:
        base['yards_gained'] = pd.to_numeric(base[c], errors='coerce')
        print(f"[info] using yards column: {c}")
        break
else:
    raise RuntimeError("No yards-gained column found (tried YardsGained, Yards, PlayResult).")

# keep only plays we actually exported
pat = re.compile(r'play_(\d+)_tile\.png')
have = set()
if os.path.isdir(IMG_DIR):
    for fn in os.listdir(IMG_DIR):
        m = pat.match(fn)
        if m:
            have.add(int(m.group(1)))
else:
    print(f"[warn] IMG_DIR does not exist: {IMG_DIR}")

print(f"[info] found {len(have)} image files in {IMG_DIR}")

labels = base[['PlayId', 'yards_gained']].copy()
labels = labels[labels['PlayId'].isin(have)]

# NEW: just the image filename, no path
labels['img_name'] = labels['PlayId'].apply(
    lambda pid: f"play_{int(pid)}_tile.png"
)

out_csv = os.path.join(IMG_DIR, 'labels.csv')
labels.to_csv(out_csv, index=False)
print("Wrote:", out_csv, "rows:", len(labels))
if not labels.empty:
    print("[info] sample row:")
    print(labels.iloc[0])
