# CNN training for yardage prediction

Quick instructions to train the CNN on `out_all_plays` images.

1. Install dependencies (prefer a venv):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run training (example):

```bash
python3 scripts/train_cnn.py --labels data/labels_small.csv --images out_all_plays --epochs 10 --batch-size 64 --pretrained
```

Notes:
- Progress bars from `tqdm` show per-batch progress for train and validation.
- Use `--workers` to increase data loading workers (e.g., `--workers 8`) if your machine has CPU cores.
- For best speed, run on a CUDA GPU (script auto-detects `cuda` if available). To force CPU: `--device cpu`.
- The script saves checkpoints in `checkpoints/` and `best_model.pth` for the lowest validation loss.

If you want a faster but lower-capacity model, modify `make_model` in `scripts/train_cnn.py` to use `mobilenet_v3_small` or a smaller backbone.
