"""Canonical, single-file trainer for the project.

Features:
- Accepts `--weights-path` to load local .pth weights safely
- `--freeze-backbone` to only train the final head
- Works on CPU / CUDA / MPS
"""
from pathlib import Path
import argparse
import csv
import os
import sys
import time
import random

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from math import isfinite
from statistics import mean
import numpy as _np

from cnn.dataset import FootballDataset
from cnn.utils import get_transforms
from cnn.model import build_resnet18


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--img-dir', required=True)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=None)
    p.add_argument('--img-size', type=int, default=160)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--optim', choices=['adam', 'adamw', 'sgd'], default='adamw', help='optimizer')
    p.add_argument('--save-dir', default='checkpoints/run_fresh')
    p.add_argument('--resume', default=None)
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--weights-path', default=None, help='local .pth file to load weights from')
    p.add_argument('--freeze-backbone', action='store_true', help='freeze backbone parameters (only train head)')
    p.add_argument('--backbone', choices=['resnet18', 'resnet34', 'resnet50'], default='resnet18', help='backbone model')
    p.add_argument('--loss', choices=['smooth_l1', 'l1', 'huber'], default='smooth_l1', help='loss for regression')
    p.add_argument('--scheduler', choices=['none', 'cosine', 'plateau', 'onecycle'], default='plateau', help='LR scheduler')
    p.add_argument('--unfreeze-epoch', type=int, default=0, help='epoch to unfreeze backbone (0=no unfreeze)')
    p.add_argument('--early-stop-patience', type=int, default=0, help='stop if no improvement after this many epochs (0=no early stop)')
    p.add_argument('--target-normalize', action='store_true', help='normalize targets (subtract mean / divide std) based on training split')
    p.add_argument('--accuracy-threshold', type=float, default=3.0, help='yard threshold used to compute accuracy (percent within threshold)')
    p.add_argument('--mixup-alpha', type=float, default=0.0, help='mixup alpha for training (0=no mixup)')
    p.add_argument('--grad-clip', type=float, default=0.0, help='clip gradients to this value (0=no clip)')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--monitor', choices=['mae', 'accuracy'], default='mae', help='metric to monitor for best checkpoint')
    return p.parse_args()


class TransformedSubset(torch.utils.data.Dataset):
    """Wrap a Subset to apply transforms on the fly (pickle-safe)."""
    def __init__(self, subset, transform=None, target_mean=0.0, target_std=1.0):
        self.subset = subset
        self.transform = transform
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        # normalize label if requested
        if isinstance(label, torch.Tensor):
            lbl = label
        else:
            lbl = torch.tensor(float(label), dtype=torch.float32)
        if self.target_std and self.target_std != 1.0:
            lbl = (lbl - float(self.target_mean)) / float(self.target_std)
        return img, lbl


def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    mps = getattr(torch.backends, 'mps', None)
    if mps and getattr(mps, 'is_available', lambda: False)():
        return torch.device('mps')
    return torch.device('cpu')


def load_weights_local(model, weights_path, map_location='cpu'):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"weights file not found: {weights_path}")
    state = torch.load(weights_path, map_location=map_location)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state_dict = state['model_state_dict']
    else:
        state_dict = state
    new_state = {}
    for k, v in state_dict.items():
        nk = k.replace('module.', '') if k.startswith('module.') else k
        new_state[nk] = v
    model_state = model.state_dict()
    filtered = {}
    for k, v in new_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            # skip mismatched keys (e.g., fc classifier for 1000 classes)
            if k in model_state:
                print(f"Skipping weight '{k}' due to shape mismatch: {v.shape} vs {model_state[k].shape}")
    model.load_state_dict(filtered, strict=False)


def build_dataloaders(csv_path, img_dir, img_size, batch_size, num_workers, normalize_targets=False):
    # simple 80/20 random split
    ds = FootballDataset(csv_path, img_dir, transform=None)
    n = len(ds)
    if n == 0:
        raise RuntimeError('Empty dataset')
    train_n = int(0.8 * n)
    val_n = n - train_n
    train_subset, val_subset = random_split(ds, [train_n, val_n])
    train_tf = get_transforms(True, img_size)
    val_tf = get_transforms(False, img_size)
    # compute target stats on training subset if requested
    target_mean = 0.0
    target_std = 1.0
    if normalize_targets:
        # subset is torch.utils.data.Subset with .dataset and .indices
        try:
            df = train_subset.dataset.df
            ycol = train_subset.dataset.y_col
            vals = df.iloc[train_subset.indices][ycol].astype(float)
            target_mean = float(vals.mean())
            target_std = float(vals.std()) if float(vals.std()) > 0 else 1.0
        except Exception:
            # fallback: compute by iterating
            ys = [float(train_subset.dataset[i][1]) for i in train_subset.indices]
            import numpy as _np
            target_mean = float(_np.mean(ys))
            target_std = float(_np.std(ys)) if float(_np.std(ys)) > 0 else 1.0

    train_ds = TransformedSubset(train_subset, transform=train_tf, target_mean=target_mean, target_std=target_std)
    val_ds = TransformedSubset(val_subset, transform=val_tf, target_mean=target_mean, target_std=target_std)
    return train_ds, val_ds


def collate_metrics(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_mae', 'val_rmse', 'val_r2', 'val_pearson', 'within1', 'within3', 'within5', 'accuracy', 'timestamp'])


def train_loop(model, train_loader, device, optimizer, loss_fn, scaler, use_amp, scheduler=None, mixup_alpha=0.0, grad_clip=0.0):
    model.train()
    running_loss = 0.0
    n = 0
    pbar = tqdm(train_loader, desc='train', leave=False)
    for xb, yb in pbar:
        xb = xb.to(device)
        yb = yb.to(device).unsqueeze(1).float()
        optimizer.zero_grad()
        # mixup augmentation for regression
        if mixup_alpha and mixup_alpha > 0.0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            batch_size = xb.size(0)
            index = torch.randperm(batch_size).to(device)
            xb = lam * xb + (1 - lam) * xb[index]
            yb = lam * yb + (1 - lam) * yb[index]
        if use_amp:
            with torch.amp.autocast(device_type=device.type):
                preds = model(xb)
                loss = loss_fn(preds, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            if grad_clip and grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        # step per-batch scheduler (e.g., OneCycleLR)
        if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            try:
                scheduler.step()
            except Exception:
                pass
        running_loss += loss.item() * xb.size(0)
        n += xb.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return running_loss / max(1, n)


def validate(model, loader, device, loss_fn, use_amp, accuracy_threshold=3.0):
    model.eval()
    total_loss = 0.0
    n = 0
    sum_abs = 0.0
    sum_sq = 0.0
    preds_all = []
    targets_all = []
    with torch.no_grad():
        pbar = tqdm(loader, desc='val', leave=False)
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1).float()
            if use_amp:
                with torch.amp.autocast(device_type=device.type):
                    preds = model(xb)
                    loss = loss_fn(preds, yb)
            else:
                preds = model(xb)
                loss = loss_fn(preds, yb)
            total_loss += loss.item() * xb.size(0)
            err = (preds - yb).detach().cpu()
            sum_abs += err.abs().sum().item()
            sum_sq += (err ** 2).sum().item()
            n += xb.size(0)
            preds_all.append(preds.detach().cpu().numpy().reshape(-1))
            targets_all.append(yb.detach().cpu().numpy().reshape(-1))
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    preds_all = _np.concatenate(preds_all) if preds_all else _np.array([])
    targets_all = _np.concatenate(targets_all) if targets_all else _np.array([])
    # denormalize predictions/targets if loader.dataset provides stats
    try:
        ds = loader.dataset
        tgt_mean = getattr(ds, 'target_mean', 0.0)
        tgt_std = getattr(ds, 'target_std', 1.0)
        if tgt_std is None:
            tgt_std = 1.0
        if preds_all.size and (tgt_std != 1.0 or tgt_mean != 0.0):
            preds_all = preds_all * float(tgt_std) + float(tgt_mean)
            targets_all = targets_all * float(tgt_std) + float(tgt_mean)
    except Exception:
        pass
    mae = sum_abs / max(1, n)
    rmse = (sum_sq / max(1, n)) ** 0.5

    # additional metrics: R2, Pearson corr, within thresholds
    r2 = None
    pearson = None
    within1 = within3 = within5 = None
    accuracy = None
    try:
        if preds_all.size and targets_all.size:
            ss_res = _np.sum((targets_all - preds_all) ** 2)
            ss_tot = _np.sum((targets_all - targets_all.mean()) ** 2)
            r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
            # pearson
            if preds_all.std() > 0 and targets_all.std() > 0:
                pearson = float(_np.corrcoef(preds_all, targets_all)[0, 1])
            within1 = float(_np.mean(_np.abs(preds_all - targets_all) <= 1.0))
            within3 = float(_np.mean(_np.abs(preds_all - targets_all) <= 3.0))
            within5 = float(_np.mean(_np.abs(preds_all - targets_all) <= 5.0))
            # accuracy is percent within the configured threshold
            try:
                accuracy = float(_np.mean(_np.abs(preds_all - targets_all) <= float(accuracy_threshold)))
            except Exception:
                accuracy = None
    except Exception:
        pass

    return total_loss / max(1, n), mae, rmse, r2, pearson, within1, within3, within5, accuracy


def main():
    args = parse_args()
    # device selection
    device = select_device()

    # determine num_workers default
    if args.num_workers is None:
        if sys.platform == 'darwin':
            num_workers = 0
        else:
            num_workers = min(4, max(1, (os.cpu_count() or 1) // 2))
    else:
        num_workers = args.num_workers

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_ds, val_ds = build_dataloaders(args.csv, args.img_dir, args.img_size, args.batch_size, num_workers, normalize_targets=args.target_normalize)

    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=max(0, num_workers), pin_memory=pin_memory)

    # build chosen backbone
    if args.backbone == 'resnet18':
        model = build_resnet18(pretrained=args.pretrained)
    else:
        # lazy import for other backbones
        from torchvision import models
        if args.backbone == 'resnet34':
            m = models.resnet34(pretrained=args.pretrained)
        else:
            m = models.resnet50(pretrained=args.pretrained)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 1)
        model = m
    # attempt to load local weights if provided
    if args.weights_path:
        try:
            load_weights_local(model, args.weights_path, map_location='cpu')
            print(f"Loaded weights from {args.weights_path}")
        except Exception as e:
            print(f"Warning: failed to load weights from {args.weights_path}: {e}")

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        print('Backbone frozen; training only head parameters')

    model.to(device)

    if args.optim == 'adamw':
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    # loss selection
    if args.loss == 'l1':
        loss_fn = nn.L1Loss()
    elif args.loss == 'huber':
        loss_fn = nn.SmoothL1Loss(beta=1.0)
    else:
        loss_fn = nn.SmoothL1Loss()

    # scheduler placeholder (OneCycle created after train_loader)
    scheduler = None

    use_amp = args.amp and device.type in ('cuda', 'mps')
    scaler = torch.amp.GradScaler() if use_amp else None

    os.makedirs(args.save_dir, exist_ok=True)
    metrics_path = Path(args.save_dir) / 'metrics.csv'
    collate_metrics(str(metrics_path))

    # choose initial best score depending on monitored metric
    if args.monitor == 'mae':
        best_score = float('inf')
    else:
        best_score = -float('inf')
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # handle unfreeze scheduling: unfreeze backbone at specified epoch
        if args.unfreeze_epoch and epoch == args.unfreeze_epoch:
            for name, param in model.named_parameters():
                param.requires_grad = True
            optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
            print(f"Unfroze backbone at epoch {epoch}; optimizer re-created with trainable params")
        # if OneCycleLR requested, create scheduler now (requires len(train_loader))
        if args.scheduler == 'onecycle' and scheduler is None:
            try:
                scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
            except Exception as e:
                print('Warning: could not create OneCycleLR:', e)

        train_loss = train_loop(model, train_loader, device, optimizer, loss_fn, scaler, use_amp, scheduler=scheduler, mixup_alpha=args.mixup_alpha, grad_clip=args.grad_clip)
        val_loss, val_mae, val_rmse, val_r2, val_pearson, within1, within3, within5, accuracy = validate(model, val_loader, device, loss_fn, use_amp, args.accuracy_threshold)
        now = time.strftime('%Y-%m-%d %H:%M:%S')
        secs = time.time() - t0
        samples_per_sec = int(len(train_loader.dataset) / max(1e-6, secs))
        # current LR
        try:
            lr = optimizer.param_groups[0]['lr']
        except Exception:
            lr = 0.0
        print(f"Epoch {epoch}/{args.epochs}, LR: {lr:.2e}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}, R2: {val_r2 if val_r2 is not None else 'n/a'}, Pearson: {val_pearson if val_pearson is not None else 'n/a'}, within1: {within1:.3f} within3: {within3:.3f} within5: {within5:.3f}, Accuracy(@{args.accuracy_threshold}): {accuracy:.3f}, samples/sec: {samples_per_sec}")
        with open(metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_mae, val_rmse, val_r2, val_pearson, within1, within3, within5, accuracy, now])
        last_ckpt = Path(args.save_dir) / 'last_checkpoint.pth'
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, str(last_ckpt))
        # decide whether this epoch improved according to the chosen monitor
        improved = False
        if args.monitor == 'mae':
            if val_mae < best_score:
                improved = True
                best_score = val_mae
        else:  # monitor == 'accuracy'
            if accuracy is not None and accuracy > best_score:
                improved = True
                best_score = accuracy

        if improved:
            best_ckpt = Path(args.save_dir) / 'best_checkpoint.pth'
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, str(best_ckpt))
            no_improve = 0
        else:
            no_improve += 1

        # step scheduler (per-epoch schedulers)
        if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            try:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            except Exception:
                pass

        # early stopping
        if args.early_stop_patience and no_improve >= args.early_stop_patience:
            print(f"Early stopping: no improvement for {no_improve} epochs (patience={args.early_stop_patience})")
            break


if __name__ == '__main__':
    main()

