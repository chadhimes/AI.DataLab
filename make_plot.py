import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge

# Accuracy values
acc_exact = 10.6
acc_1 = 21.1
acc_3 = 58.8
acc_5 = 79.3

# Ring boundaries (inner radius, outer radius)
rings = [
    ("Exact", acc_exact, 0.00, 0.40, "#d73027"),
    ("±1",    acc_1,     0.40, 0.80, "#fc8d59"),
    ("±3",    acc_3,     0.80, 1.25, "#fee090"),
    ("±5",    acc_5,     1.25, 1.70, "#e0f3f8")
]

fig, ax = plt.subplots(figsize=(8,8))

for i, (label, acc, r_in, r_out, color) in enumerate(rings):

    # Draw ring WITHOUT border
    ring = Wedge(
        center=(0,0),
        r=r_out,
        theta1=0,
        theta2=360,
        width=r_out - r_in,
        facecolor=color,
        edgecolor=None  # <-- border removed
    )
    ax.add_patch(ring)

    # --- label angles to avoid overlap ---
    if label == "±5":
        angle = 95
    elif label == "±3":
        angle = 75
    elif label == "±1":
        angle = 60
    else:  # Exact
        angle = 45

    rad = np.deg2rad(angle)

    # Midpoint of ring
    r_mid = (r_in + r_out) / 2
    x0 = r_mid * np.cos(rad)
    y0 = r_mid * np.sin(rad)

    # Point for text outside
    x1 = (r_out + 0.3) * np.cos(rad)
    y1 = (r_out + 0.3) * np.sin(rad)

    # Leader line
    ax.plot([x0, x1], [y0, y1], color='black', linewidth=1.4)

    # Label
    ax.text(
        x1 + 0.05*np.cos(rad),
        y1 + 0.05*np.sin(rad),
        f"{label}: {acc}%",
        fontsize=14,
        va='center'
    )

# Formatting
ax.set_xlim([-2.0, 2.5])
ax.set_ylim([-2.0, 2.5])
ax.set_aspect('equal')
ax.axis('off')

plt.show()
