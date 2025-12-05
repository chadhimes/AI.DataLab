#!/usr/bin/env python3
"""
Evaluate one or more checkpoints with Test-Time Augmentation (TTA) and ensembling.

Usage:
  python3 scripts/evaluate_ensemble.py --checkpoints checkpoints/best_model.pth --labels data/labels_small.csv --images out_all_plays --use-timm --timm-backbone tf_efficientnet_b4 --batch-size 32

This script loads checkpoints, runs TTA (original + horizontal flip), averages predictions per model, then averages across models.
Saves `predictions_ensemble.csv` with columns: `img_name, true_yards, pred_yards`.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models

from image_dataset import PlayImageDataset

try:
    import timm
    _HAS_TIMM = True
except Exception:
    timm = None
    _HAS_TIMM = False


def make_model(pretrained=False, device='cpu', small=False, use_timm=False, timm_backbone='tf_efficientnet_b3'):
    if use_timm and _HAS_TIMM:
        model = timm.create_model(timm_backbone, pretrained=pretrained, num_classes=1)
        return model.to(device)

    if small:
        model = models.mobilenet_v3_small(pretrained=pretrained)
    else:
        model = models.mobilenet_v3_large(pretrained=pretrained)
    try:
        in_features = model.classifier[0].in_features
    except Exception:
        in_features = 1280
    model.classifier = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True, help='Paths to checkpoint files to ensemble')
    parser.add_argument('--labels', type=str, default='data/labels_small.csv')
    parser.add_argument('--images', type=str, default='out_all_plays')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use-timm', action='store_true')
    parser.add_argument('--timm-backbone', type=str, default='tf_efficientnet_b4')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--output', type=str, default='predictions_ensemble.csv')
    args = parser.parse_args()

    device = torch.device(args.device)

    df = pd.read_csv(args.labels)
    # only keep images that exist
    df['image_path'] = df['img_name'].apply(lambda x: os.path.join(args.images, x))
    df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)

    # basic transform (no random) for evaluation; TTA will be applied manually
    base_transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

    # Create a lightweight dataset wrapper for evaluation
    class EvalDataset(torch.utils.data.Dataset):
        def __init__(self, df, transform):
            self.df = df
            self.transform = transform
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            from PIL import Image
            img = Image.open(row['image_path']).convert('RGB')
            img_t = self.transform(img)
            return img_t, float(row['yards_gained']), row['img_name']

    ds = EvalDataset(df, base_transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # TTA transforms as lambdas that accept a batch tensor and return transformed batch tensor
    tta_transforms = [lambda x: x, lambda x: torch.flip(x, dims=[3])]

    model_preds = []
    all_labels = None
    all_names = None

    for ckpt_path in args.checkpoints:
        print('Loading', ckpt_path)
        model = make_model(pretrained=False, device=device, small=args.small, use_timm=args.use_timm, timm_backbone=args.timm_backbone)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get('model_state', ckpt)
        model.load_state_dict(state)
        model.eval()

        preds = []
        labels = []
        names = []
        with torch.no_grad():
            for imgs, labs, names_batch in loader:
                imgs = imgs.to(device)
                t_preds = []
                for t in tta_transforms:
                    imgs_t = t(imgs)
                    out = model(imgs_t)
                    t_preds.append(out.detach().cpu().numpy().squeeze(1))
                t_preds = np.stack(t_preds, axis=0).mean(axis=0)
                preds.extend(t_preds.tolist())
                labels.extend([float(x) for x in labs])
                names.extend(names_batch)

        model_preds.append(np.array(preds))
        if all_labels is None:
            all_labels = np.array(labels)
            all_names = names

    ensemble_preds = np.mean(np.stack(model_preds, axis=0), axis=0)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(all_labels, ensemble_preds)
    rmse = mean_squared_error(all_labels, ensemble_preds, squared=False)
    r2 = r2_score(all_labels, ensemble_preds)
    acc5 = float((np.abs(all_labels - ensemble_preds) <= 5).sum()) / len(all_labels) * 100.0
    acc10 = float((np.abs(all_labels - ensemble_preds) <= 10).sum()) / len(all_labels) * 100.0

    print(f'Ensemble MAE: {mae:.3f}  RMSE: {rmse:.3f}  R2: {r2:.3f}  acc±5: {acc5:.1f}%  acc±10: {acc10:.1f}%')

    out_df = pd.DataFrame({'img_name': all_names, 'true_yards': all_labels, 'pred_yards': ensemble_preds})
    out_df.to_csv(args.output, index=False)
    print('Saved predictions to', args.output)


if __name__ == '__main__':
    main()
