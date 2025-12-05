import matplotlib.pyplot as plt

# ---------------------------
# Epoch numbers
# ---------------------------
epochs = list(range(1, 31))

# ---------------------------
# Train + Validation Loss
# ---------------------------
train_loss = [
    1.8823, 0.9509, 0.9170, 0.9052, 0.8970,
    0.8959, 0.8916, 0.8910, 0.8870, 0.8865,
    0.8837, 0.8822, 0.8816, 0.8788, 0.8768,
    0.8741, 0.8735, 0.8744, 0.8712, 0.8707,
    0.9371, 0.9317, 0.9319, 0.9286, 0.9273,
    0.9262, 0.9274, 0.9273, 0.9259, 0.9264
]

val_loss = [
    0.2899, 0.2858, 0.2825, 0.2797, 0.2779,
    0.2760, 0.2748, 0.2729, 0.2712, 0.2702,
    0.2690, 0.2688, 0.2682, 0.2673, 0.2668,
    0.2666, 0.2658, 0.2659, 0.2648, 0.2648,
    0.2826, 0.2827, 0.2827, 0.2825, 0.2818,
    0.2825, 0.2815, 0.2810, 0.2820, 0.2814
]

# ---------------------------
# Accuracies (exact, ±1, ±3, ±5)
# ---------------------------

acc_1 = [  # ±1 accuracy
    16.1, 17.4, 17.8, 18.1, 18.2,
    18.3, 18.2, 18.8, 19.0, 19.2,
    19.6, 19.7, 20.0, 19.9, 20.2,
    20.2, 20.8, 20.9, 21.1, 21.6,
    22.5, 23.1, 22.6, 22.9, 22.9,
    22.7, 22.9, 23.2, 23.7, 23.3
]

acc_3 = [
    48.2, 50.3, 51.2, 51.6, 52.2,
    52.5, 52.9, 54.1, 54.7, 55.1,
    55.9, 56.1, 56.4, 57.2, 57.7,
    57.5, 58.1, 58.1, 58.9, 58.9,
    60.3, 60.3, 60.6, 60.3, 60.5,
    60.0, 60.4, 60.8, 60.3, 60.9
]

acc_5 = [
    76.3, 77.6, 78.4, 78.6, 78.9,
    79.2, 79.0, 79.3, 79.5, 79.9,
    79.8, 79.9, 80.0, 80.2, 80.2,
    80.5, 80.6, 80.5, 80.6, 80.5,
    81.1, 81.0, 81.0, 81.0, 81.3,
    81.1, 81.1, 81.4, 81.0, 81.2
]

exact_acc = [
    8.0, 8.3, 8.6, 9.1, 9.5,
    9.6, 9.8, 10.2, 10.3, 10.2,
    10.4, 10.3, 10.3, 10.2, 10.1,
    10.3, 10.5, 10.5, 10.6, 10.6,
    11.4, 11.8, 11.8, 11.9, 11.5,
    11.4, 11.5, 11.4, 12.0, 11.8
]

# -------------------------------------------------------------
# PLOT 1 — Train vs Validation Loss (portrait)
# -------------------------------------------------------------
import os

# Make text and lines compact for smaller images
plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'legend.fontsize': 8
})

# Create output folder for saved plots
out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)

plt.figure(figsize=(4, 4))
plt.plot(epochs, train_loss, label="Train Loss", linewidth=1.5)
plt.plot(epochs, val_loss, label="Validation Loss", linewidth=1.5)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.grid(True, alpha=0.3)
plt.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "train_vs_val_loss_vertical.png"), dpi=150, bbox_inches='tight')
plt.show()

# -------------------------------------------------------------
# PLOT 2 — Accuracies (±1, ±3, ±5, exact) (portrait)
# -------------------------------------------------------------
plt.figure(figsize=(4, 5))
plt.plot(epochs, acc_5, label="Accuracy ±5", linewidth=1.5)
plt.plot(epochs, acc_3, label="Accuracy ±3", linewidth=1.5)
plt.plot(epochs, acc_1, label="Accuracy ±1", linewidth=1.5)
plt.plot(epochs, exact_acc, label="Exact Accuracy", linewidth=1.5)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracies Over Epochs")
plt.grid(True, alpha=0.3)
plt.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "accuracies_over_epochs_vertical.png"), dpi=150, bbox_inches='tight')
plt.show()