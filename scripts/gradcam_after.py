#!/usr/bin/env python3
"""
Wait for a checkpoint file to become stable, then run `gradcam.py` to generate heatmaps.

This helper is useful to start Grad-CAM automatically after a long training run finishes
without interrupting the training process. It watches the `--checkpoint` path and
considers the file "ready" when it exists and its size hasn't changed for
`--stability-secs` seconds.

Example:
  python3 scripts/gradcam_after.py --checkpoint checkpoints_highacc_run/best_model.pth \
    --images out_all_plays --out gradcam_after_results --threshold 0.6

If you prefer a simple check (file exists), set `--stability-secs 0`.
"""
import argparse
import time
import subprocess
from pathlib import Path
import sys


def wait_for_stable_file(path: Path, stability_secs: float = 30.0, poll_interval: float = 5.0):
    last_size = -1
    stable_since = None
    while True:
        if not path.exists():
            time.sleep(poll_interval)
            continue
        size = path.stat().st_size
        now = time.time()
        if size == last_size:
            if stable_since is None:
                stable_since = now
            elif now - stable_since >= stability_secs:
                return True
        else:
            stable_since = None
            last_size = size
        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file to wait for')
    parser.add_argument('--images', default='out_all_plays')
    parser.add_argument('--out', default='gradcam_after_results')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--use-timm', action='store_true')
    parser.add_argument('--timm-backbone', default='efficientnet_b0')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--high-out', default=None)
    parser.add_argument('--move', action='store_true')
    parser.add_argument('--stability-secs', type=float, default=30.0, help='Seconds of unchanged file size to consider checkpoint stable')
    parser.add_argument('--poll-interval', type=float, default=5.0)
    parser.add_argument('--extra-args', default='', help='Extra args to forward to gradcam.py')
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    print(f"Waiting for checkpoint {ckpt} to exist and be stable ({args.stability_secs}s)...")
    try:
        wait_for_stable_file(ckpt, stability_secs=args.stability_secs, poll_interval=args.poll_interval)
    except KeyboardInterrupt:
        print("Interrupted while waiting for checkpoint.")
        sys.exit(1)

    # build gradcam command
    cmd = [sys.executable, 'scripts/gradcam.py', '--checkpoint', str(ckpt), '--images', args.images, '--out', args.out, '--img-size', str(args.img_size), '--device', args.device]
    if args.use_timm:
        cmd += ['--use-timm', '--timm-backbone', args.timm_backbone]
    if args.small:
        cmd += ['--small']
    if args.pretrained:
        cmd += ['--pretrained']
    if args.threshold is not None:
        cmd += ['--threshold', str(args.threshold)]
    if args.high_out:
        cmd += ['--high-out', args.high_out]
    if args.move:
        cmd += ['--move']
    if args.extra_args:
        # split simple args string into list
        cmd += args.extra_args.split()

    print("Checkpoint ready — running Grad-CAM:")
    print(' '.join(cmd))
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
