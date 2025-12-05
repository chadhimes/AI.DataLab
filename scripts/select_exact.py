#!/usr/bin/env python3
"""
Select images the model predicted exactly (rounded) and copy them to a folder.

Also can optionally run `scripts/gradcam.py` on the selected images to create heatmaps.

Usage:
  python3 scripts/select_exact.py --checkpoint checkpoints_highacc_run/best_model.pth \
    --labels data/labels.csv --images out_all_plays --out exact_images --make-gradcam

This script computes predictions, denormalizes them using the labels CSV mean/std,
rounds to nearest integer, and copies images where rounded prediction == ground truth.
"""
import argparse
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
import shutil
import subprocess
import sys


def build_model(use_timm=False, timm_backbone='efficientnet_b0', small=False, pretrained=False, img_size=224):
    if use_timm:
        import timm
        model = timm.create_model(timm_backbone, pretrained=pretrained, num_classes=1)
        return model
    else:
        from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
        if small:
            m = mobilenet_v3_small(pretrained=pretrained)
        else:
            m = mobilenet_v3_large(pretrained=pretrained)
        in_features = m.classifier[0].in_features if hasattr(m.classifier[0], 'in_features') else m.classifier[-1].in_features
        m.classifier = torch.nn.Sequential(torch.nn.Linear(in_features, 1))
        return m


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # Common checkpoint key names: 'state_dict', 'model_state', 'model', 'model_state_dict'
    state = None
    if isinstance(ckpt, dict):
        for key in ('state_dict', 'model_state', 'model_state_dict', 'model_state', 'model'):
            if key in ckpt:
                state = ckpt[key]
                break
    if state is None:
        # fallback: maybe the checkpoint itself is a state dict (mapping of tensors)
        state = ckpt

    # If state is nested (e.g., contains 'model'->state_dict), attempt to unwrap
    if isinstance(state, dict) and any(not isinstance(v, torch.Tensor) for v in state.values()):
        # try to find the actual sub-dict that maps to tensors
        candidate = None
        for k, v in state.items():
            if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                candidate = v
                break
        if candidate is not None:
            state = candidate

    try:
        model.load_state_dict(state)
    except Exception:
        # Strip common prefixes (module., model., encoder.) and try again
        new_state = {}
        for k, v in state.items():
            new_key = k
            for prefix in ('module.', 'model.', 'encoder.'):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break
            new_state[new_key] = v
        model.load_state_dict(new_state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--images', default='out_all_plays')
    parser.add_argument('--out', default='exact_images')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--use-timm', action='store_true')
    parser.add_argument('--timm-backbone', default='efficientnet_b0')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--make-gradcam', action='store_true', help='Run gradcam on selected images after copying')
    parser.add_argument('--gradcam-args', default='', help='Extra args forwarded to gradcam.py')
    args = parser.parse_args()

    # Prefer explicit device selection: support 'cpu', 'cuda', and 'mps' (Metal)
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    df = pd.read_csv(args.labels)
    # support both 'img_name' and 'img'
    if 'img_name' not in df.columns and 'img' in df.columns:
        df = df.rename(columns={'img': 'img_name'})
    if 'img_name' not in df.columns:
        raise ValueError('labels csv must contain img_name column')
    if 'yards_gained' not in df.columns:
        raise ValueError('labels csv must contain yards_gained column')

    # filter only files that exist in images dir
    df['image_path'] = df['img_name'].apply(lambda x: str(Path(args.images) / x))
    df = df[df['image_path'].apply(lambda p: Path(p).exists())].reset_index(drop=True)

    label_mean = float(df['yards_gained'].mean())
    label_std = float(df['yards_gained'].std()) if float(df['yards_gained'].std()) > 0 else 1.0

    model = build_model(use_timm=args.use_timm, timm_backbone=args.timm_backbone, small=args.small, pretrained=args.pretrained, img_size=args.img_size)
    load_checkpoint(model, args.checkpoint)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    out_dir = Path(args.out)
    exact_dir = out_dir / 'exact_matches'
    out_dir.mkdir(parents=True, exist_ok=True)
    exact_dir.mkdir(parents=True, exist_ok=True)

    exact_list = []
    for _, row in df.iterrows():
        img_path = Path(row['image_path'])
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            continue
        inp = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(inp)
        if isinstance(pred, torch.Tensor):
            pred = pred.squeeze().cpu().item()
        # denormalize
        denorm = pred * label_std + label_mean
        pred_round = int(np.rint(denorm))
        true_label = int(np.rint(float(row['yards_gained'])))
        if pred_round == true_label:
            target = exact_dir / img_path.name
            shutil.copy2(img_path, target)
            exact_list.append({'img_name': img_path.name, 'pred': denorm, 'label': true_label})

    # save CSV of exact matches
    if exact_list:
        out_csv = out_dir / 'exact_matches.csv'
        pd.DataFrame(exact_list).to_csv(out_csv, index=False)

    print(f"Copied {len(exact_list)} exact-match images to {exact_dir}")

    if args.make_gradcam and len(exact_list) > 0:
        cmd = [sys.executable, 'scripts/gradcam.py', '--checkpoint', args.checkpoint, '--images', str(exact_dir), '--out', str(out_dir / 'gradcam_exact'), '--img-size', str(args.img_size), '--device', args.device]
        if args.use_timm:
            cmd += ['--use-timm', '--timm-backbone', args.timm_backbone]
        if args.small:
            cmd += ['--small']
        if args.pretrained:
            cmd += ['--pretrained']
        if args.gradcam_args:
            cmd += args.gradcam_args.split()
        print('Running Grad-CAM on exact matches:')
        print(' '.join(cmd))
        subprocess.run(cmd)


if __name__ == '__main__':
    main()
