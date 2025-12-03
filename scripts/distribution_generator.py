import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns

# Script: scripts/comparison.py
# Purpose: compute and save an accurate distribution of `yards_gained` from
#          out_all_plays/labels.csv and produce summary statistics + plots.

LABELS_PATH = os.path.join("out_all_plays", "labels.csv")
OUT_DIR = os.path.join("out_all_plays")
FIG_PATH = os.path.join(OUT_DIR, "yards_gained_distribution.png")
CSV_OUT_PATH = os.path.join(OUT_DIR, "yards_gained_counts.csv")

print("Current working directory:", os.getcwd())
print(f"Reading labels from: {LABELS_PATH}")

# --- load ---
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"labels.csv not found at {LABELS_PATH}")

# Read only necessary columns if present
try:
    df = pd.read_csv(LABELS_PATH, usecols=["playid", "yards_gained", "png_name"])
except Exception:
    # fallback: read all and rely on column names
    df = pd.read_csv(LABELS_PATH)

if "yards_gained" not in df.columns:
    raise ValueError("labels.csv must contain a 'yards_gained' column")

# Clean and coerce to numeric
y = pd.to_numeric(df["yards_gained"], errors="coerce")
y = y.dropna()

# If yards are floats but actually integer-valued, round to nearest int
if not np.all(np.isclose(y, np.round(y))):
    # keep as floats and bin; but warn user
    print("Warning: some 'yards_gained' values are non-integer; results will be binned.")
    is_integer_like = False
else:
    is_integer_like = True
    y = y.round().astype(int)

n = len(y)
print(f"Loaded {n} non-null yards_gained values")
if n == 0:
    raise ValueError("No valid yards_gained values found in labels.csv")

# --- distribution counts ---
if is_integer_like:
    min_y, max_y = int(y.min()), int(y.max())
    # include full integer range so zeros show up
    index = np.arange(min_y, max_y + 1)
    counts = y.value_counts().reindex(index, fill_value=0).sort_index()
    probs = counts / counts.sum()
    cdf = probs.cumsum()
    out_df = pd.DataFrame({"yards": index, "count": counts.values, "prob": probs.values, "cdf": cdf.values})
else:
    # bin floats into integer bins (floor) to get approximate integer distribution
    bins = np.arange(np.floor(y.min()), np.ceil(y.max()) + 1)
    counts, edges = np.histogram(y, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    probs = counts / counts.sum()
    cdf = np.cumsum(probs)
    out_df = pd.DataFrame({"yards": centers, "count": counts, "prob": probs, "cdf": cdf})

# Save CSV of counts
out_df.to_csv(CSV_OUT_PATH, index=False)
print(f"Saved counts CSV to: {CSV_OUT_PATH}")

# --- summary stats ---
mean = y.mean()
median = y.median()
std = y.std()
quantiles = y.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.99])
minv, maxv = y.min(), y.max()

print("Summary statistics for yards_gained:")
print(f"  n = {n}")
print(f"  mean = {mean:.3f}")
print(f"  median = {median:.3f}")
print(f"  std = {std:.3f}")
print(f"  min = {minv}")
print(f"  max = {maxv}")
print("  quantiles:")
for q, val in quantiles.items():
    print(f"    {q:.2%}: {val}")

# --- plotting: simpler bar + overlayed distribution line ---
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 6))

# Bar plot of probability mass (PMF) on primary y-axis (density/probability)
if is_integer_like:
    bar_width = 0.8
else:
    if len(out_df['yards']) > 1:
        bar_width = float(out_df['yards'].iloc[1] - out_df['yards'].iloc[0]) * 0.9
    else:
        bar_width = 0.8

bars = ax.bar(out_df['yards'], out_df['prob'], color="C0", alpha=0.8, width=bar_width, label='PMF (probability)')
ax.set_xlabel('Yards Gained')
ax.set_ylabel('Probability')
ax.set_title('Distribution of Yards Gained')

# Overlay: KDE (density) on the same primary axis so both are in probability units
kd_x = None
kd_density = None
try:
    from scipy.stats import gaussian_kde
    SMOOTH_FACTOR = 1.2
    kde = gaussian_kde(y, bw_method=lambda kde_obj: kde_obj.covariance_factor() * SMOOTH_FACTOR)
    kd_x = np.linspace(float(out_df['yards'].min()), float(out_df['yards'].max()), 2000)
    kd_density = kde(kd_x)
    ax.plot(kd_x, kd_density, color='C1', lw=2, label=f'KDE (bw×{SMOOTH_FACTOR})')
except Exception:
    # Fallback: plot PMF line (probabilities)
    ax.plot(out_df['yards'], out_df['prob'], color='C1', lw=2, label='PMF (probability)')

# Build legend: PMF bars and KDE line (legend enlarged 150%)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
handles = [Patch(color='C0'), Line2D([0], [0], color='C1', lw=2)]
labels = [f'PMF (n={n})', 'KDE (density)']
base_leg = plt.rcParams.get('legend.fontsize', 10)
# make legend a little smaller than previous (was 150%) — use 110% of base
ax.legend(handles, labels, loc='upper right', prop={'size': base_leg * 1.1})

# Stats and grid: set y-grid major every 0.025 (probability)
major_locator = MultipleLocator(0.025)
ax.yaxis.set_major_locator(major_locator)
ax.grid(which='major', axis='y', linestyle='--', alpha=0.7)
ax.grid(False, axis='x')

# Add a compact stats textbox on the plot (bottom-right)
q1 = float(quantiles.loc[0.25])
q3 = float(quantiles.loc[0.75])
mean_v = float(mean)
median_v = float(median)
stats_text = (
    f"n = {n}\nmean = {mean_v:.2f}\nmedian = {median_v:.2f}\nQ1 = {q1:.2f}\nQ3 = {q3:.2f}\nstd = {std:.2f}"
)
ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.tight_layout()
plt.savefig(FIG_PATH, dpi=150)
print(f"Saved distribution figure to: {FIG_PATH}")
plt.show()

print("Done.")
