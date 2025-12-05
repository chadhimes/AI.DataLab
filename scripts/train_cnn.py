#!/usr/bin/env python3
"""
Training script for predicting `yards_gained` from play images.

Features:
- Optional `timm` backbones (EfficientNet) for high accuracy
- AdamW optimizer with weight decay
- MixUp augmentation for regression
- Warmup + cosine LR scheduler (optional)
- AMP (automatic mixed precision) when CUDA is available
- Progress bars and terminal metrics including MAE, RMSE, R2, and percent within ±1/±3/±5 yards
"""

import os
import argparse
from pathlib import Path
import time
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import torchvision.models as models
from tqdm import tqdm

from image_dataset import PlayImageDataset
from copy import deepcopy
import warnings
import torch.nn.functional as F

# optional albumentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALB = True
except Exception:
    A = None
    ToTensorV2 = None
    _HAS_ALB = False

# optional timm
try:
    import timm
    _HAS_TIMM = True
except Exception:
    timm = None
    _HAS_TIMM = False


def make_model(pretrained=True, device='cpu', small=False, use_timm=False, timm_backbone='tf_efficientnet_b3'):
    if use_timm and _HAS_TIMM:
        try:
            model = timm.create_model(timm_backbone, pretrained=pretrained, num_classes=1)
        except Exception:
            model = timm.create_model('tf_efficientnet_b3', pretrained=pretrained, num_classes=1)
        return model.to(device)

    if small:
        model = models.mobilenet_v3_small(pretrained=pretrained)
    else:
        model = models.mobilenet_v3_large(pretrained=pretrained)

    try:
        in_features = model.classifier[0].in_features
    except Exception:
        in_features = 1280

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, 1)
    )
    return model.to(device)


class HybridModel(nn.Module):
    """Wrap a backbone that returns feature vectors and add a regression and classification head.

    NOTE: For robustness this wrapper currently expects the provided `backbone` to return
    a feature tensor of shape (B, feat_dim) when called (i.e., its classifier returns
    features instead of final predictions). The script uses this with `--small` (mobilenet_v3_small)
    which is adjusted below when `--hybrid` is used.
    """
    def __init__(self, backbone, feat_dim, n_bins=65, min_bin=-32):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.n_bins = int(n_bins)
        self.min_bin = int(min_bin)
        # regression head
        self.reg_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        # classification head (logits)
        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, self.n_bins)
        )

    def forward(self, x):
        # assume backbone(x) returns feature vectors
        feats = self.backbone(x)
        # if backbone returns a tensor with extra dims, flatten
        if feats.dim() > 2:
            feats = feats.view(feats.size(0), -1)
        reg = self.reg_head(feats)
        logits = self.cls_head(feats)
        return reg, logits


class ModelEMA:
    """Simple exponential moving average (EMA) of model weights."""
    def __init__(self, model, decay=0.9999, device=None):
        self.decay = decay
        self.device = device
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad = False
        if device is not None:
            self.ema.to(device)

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= self.decay
                    v += (1.0 - self.decay) * msd[k].detach().to(v.device)

    def state_dict(self):
        return self.ema.state_dict()



def mixup_data(x, y, alpha=0.2, device='cpu'):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def rand_bbox(size, lam):
    # size: (batch, channels, H, W)
    _, _, H, W = size
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    # uniform center
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return y1, x1, y2, x2


def cutmix_data(x, y, alpha=1.0, device='cpu'):
    """Apply CutMix to a batch (regression labels are mixed proportionally)."""
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    y1, x1, y2, x2 = rand_bbox(x.size(), lam)
    # mix images
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    # recompute lambda as pixel area ratio
    box_area = float((y2 - y1) * (x2 - x1))
    lam = 1.0 - box_area / float(x.size(2) * x.size(3))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def train_epoch(model, loader, optimizer, criterion, device, label_mean, label_std, use_amp=False, mixup_alpha=0.0, cutmix_prob=0.0, cutmix_alpha=1.0, accumulate_steps=1):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    pbar = tqdm(loader, desc='Train', leave=False)

    for imgs, labels in pbar:
        if isinstance(imgs, list):
            imgs = torch.stack(imgs).to(device)
        else:
            imgs = imgs.to(device)
        labels = torch.as_tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)

        # Possibly apply CutMix (preferred) or MixUp if configured
        # Hybrid models don't currently support mixing for the classification target,
        # so skip MixUp/CutMix when using a HybridModel to avoid ambiguous soft labels.
        if isinstance(model, HybridModel):
            labels_a, labels_b, lam = labels, None, 1.0
        else:
            do_cutmix = (cutmix_prob > 0.0) and (random.random() < float(cutmix_prob))
            if do_cutmix:
                imgs, labels_a, labels_b, lam = cutmix_data(imgs, labels, alpha=cutmix_alpha, device=device)
            elif mixup_alpha > 0.0 and (random.random() < 0.5):
                imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, alpha=mixup_alpha, device=device)
            else:
                labels_a, labels_b, lam = labels, None, 1.0

        # gradient accumulation: scale loss and step only every `accumulate_steps` batches
        if (not use_amp):
            optimizer.zero_grad()
        else:
            # with AMP we'll defer zero_grad until stepping to be safe
            pass
        if use_amp:
            with torch.cuda.amp.autocast():
                preds = model(imgs)
                # hybrid model returns (regression, logits)
                if isinstance(preds, tuple):
                    reg_preds, logits = preds
                    # regression loss
                    if lam != 1.0 and labels_b is not None:
                        reg_loss = lam * criterion(reg_preds, labels_a) + (1 - lam) * criterion(reg_preds, labels_b)
                    else:
                        reg_loss = criterion(reg_preds, labels_a)
                    # classification loss (use rounded integer bins)
                    try:
                        cls_targets = torch.clamp(torch.round(labels_a).squeeze(1).long() - model.min_bin, 0, model.n_bins - 1).to(logits.device)
                    except Exception:
                        cls_targets = torch.clamp(torch.round(labels_a).squeeze(1).long(), 0, model.n_bins - 1).to(logits.device)
                    # optional focal loss for classification head
                    focal_alpha = float(getattr(model, 'focal_alpha', 0.0))
                    focal_gamma = float(getattr(model, 'focal_gamma', 0.0))
                    if focal_gamma > 0.0:
                        logp = F.log_softmax(logits, dim=1)
                        p = logp.exp()
                        pt = p.gather(1, cls_targets.view(-1, 1)).squeeze(1)
                        ce = F.nll_loss(logp, cls_targets, reduction='none')
                        focal = ((1 - pt) ** focal_gamma) * ce
                        if focal_alpha > 0.0:
                            focal = focal_alpha * focal
                        cls_loss = focal.mean()
                    else:
                        cls_loss = F.cross_entropy(logits, cls_targets)
                    loss = reg_loss + float(getattr(model, 'class_weight', 1.0)) * cls_loss
                    # use regression preds for downstream metrics
                    preds = reg_preds
                else:
                    if lam != 1.0 and labels_b is not None:
                        loss = lam * criterion(preds, labels_a) + (1 - lam) * criterion(preds, labels_b)
                    else:
                        loss = criterion(preds, labels_a)
            # scale loss by accumulate steps to keep lr/grad scale consistent
            loss = loss / float(max(1, accumulate_steps))
            scaler.scale(loss).backward()
            step = getattr(train_epoch, '_step', 0) + 1
            setattr(train_epoch, '_step', step)
            if step % accumulate_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            preds = model(imgs)
            if isinstance(preds, tuple):
                reg_preds, logits = preds
                if lam != 1.0 and labels_b is not None:
                    reg_loss = lam * criterion(reg_preds, labels_a) + (1 - lam) * criterion(reg_preds, labels_b)
                else:
                    reg_loss = criterion(reg_preds, labels_a)
                try:
                    cls_targets = torch.clamp(torch.round(labels_a).squeeze(1).long() - model.min_bin, 0, model.n_bins - 1).to(logits.device)
                except Exception:
                    cls_targets = torch.clamp(torch.round(labels_a).squeeze(1).long(), 0, model.n_bins - 1).to(logits.device)
                focal_alpha = float(getattr(model, 'focal_alpha', 0.0))
                focal_gamma = float(getattr(model, 'focal_gamma', 0.0))
                if focal_gamma > 0.0:
                    logp = F.log_softmax(logits, dim=1)
                    p = logp.exp()
                    pt = p.gather(1, cls_targets.view(-1, 1)).squeeze(1)
                    ce = F.nll_loss(logp, cls_targets, reduction='none')
                    focal = ((1 - pt) ** focal_gamma) * ce
                    if focal_alpha > 0.0:
                        focal = focal_alpha * focal
                    cls_loss = focal.mean()
                else:
                    cls_loss = F.cross_entropy(logits, cls_targets)
                loss = reg_loss + float(getattr(model, 'class_weight', 1.0)) * cls_loss
                # use regression preds for downstream metrics
                preds = reg_preds
            else:
                if lam != 1.0 and labels_b is not None:
                    loss = lam * criterion(preds, labels_a) + (1 - lam) * criterion(preds, labels_b)
                else:
                    loss = criterion(preds, labels_a)
            loss = loss / float(max(1, accumulate_steps))
            loss.backward()
            step = getattr(train_epoch, '_step', 0) + 1
            setattr(train_epoch, '_step', step)
            if step % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # update EMA if present on optimizer object (we attach ema to optimizer._ema)
        try:
            ema = getattr(optimizer, '_ema', None)
            if ema is not None:
                ema.update(model)
        except Exception:
            pass

        # record running loss with unscaled loss per-sample
        running_loss += (loss.item() * float(max(1, accumulate_steps))) * imgs.size(0)

        preds_denorm = preds.detach().cpu().numpy() * label_std + label_mean
        if lam != 1.0 and labels_b is not None:
            labels_denorm = (lam * labels_a.detach().cpu().numpy() + (1 - lam) * labels_b.detach().cpu().numpy()) * label_std + label_mean
        else:
            labels_denorm = labels.detach().cpu().numpy() * label_std + label_mean

        batch_mae = float((np.abs(preds_denorm - labels_denorm)).mean())
        running_mae += batch_mae * imgs.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mae': f'{batch_mae:.3f}'})

    return running_loss / len(loader.dataset), running_mae / len(loader.dataset)


def validate_epoch(model, loader, criterion, device, label_mean, label_std, tta=0):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    running_mse = 0.0
    preds_all = []
    preds_cls_all = []
    labels_all = []
    pbar = tqdm(loader, desc='Val  ', leave=False)
    with torch.no_grad():
        for imgs, labels in pbar:
            if isinstance(imgs, list):
                imgs = torch.stack(imgs).to(device)
            else:
                imgs = imgs.to(device)
            labels = torch.as_tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)

            out = model(imgs)
            # handle hybrid tuple output
            if isinstance(out, tuple):
                reg_preds, logits = out
                # simple TTA by horizontally flipping images and averaging predictions
                if tta and tta > 0:
                    flipped = torch.flip(imgs, dims=[3])
                    with torch.no_grad():
                        reg2, logits2 = model(flipped)
                    reg_preds = (reg_preds + reg2) / 2.0
                    logits = (logits + logits2) / 2.0
                loss = criterion(reg_preds, labels)
                running_loss += loss.item() * imgs.size(0)

                preds_denorm = reg_preds.detach().cpu().numpy() * label_std + label_mean
                labels_denorm = labels.detach().cpu().numpy() * label_std + label_mean
                # classification integer prediction from logits
                try:
                    probs = torch.softmax(logits, dim=1)
                    cls_inds = torch.argmax(probs, dim=1).detach().cpu().numpy()
                    cls_vals = (cls_inds + model.min_bin).tolist()
                    # confidence-based snapping: if confident, snap regression to cls bin
                    snap_thresh = float(getattr(model, 'snap_threshold', 0.0))
                    if snap_thresh > 0.0:
                        top_probs, _ = probs.max(dim=1)
                        snap_mask = (top_probs > snap_thresh).detach().cpu().numpy().astype(bool)
                        if np.any(snap_mask):
                            base_arr = preds_denorm.flatten()
                            snapped = np.array(cls_vals, dtype=float)
                            # ensure shapes align (batch,)
                            if snapped.shape != base_arr.shape:
                                snapped = snapped.reshape(base_arr.shape)
                            preds_denorm = np.where(snap_mask, snapped, base_arr)
                except Exception:
                    cls_vals = [0] * imgs.size(0)
                batch_mae = float((np.abs(preds_denorm - labels_denorm)).mean())
                batch_mse = float(((preds_denorm - labels_denorm) ** 2).mean())
                running_mae += batch_mae * imgs.size(0)
                running_mse += batch_mse * imgs.size(0)
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mae': f'{batch_mae:.3f}'})

                preds_all.extend(preds_denorm.flatten().tolist())
                preds_cls_all.extend(cls_vals)
                labels_all.extend(labels_denorm.flatten().tolist())
            else:
                preds = out
                # simple TTA by horizontally flipping images and averaging predictions
                if tta and tta > 0:
                    flipped = torch.flip(imgs, dims=[3])
                    with torch.no_grad():
                        preds += model(flipped)
                    preds = preds / 2.0
                loss = criterion(preds, labels)
                running_loss += loss.item() * imgs.size(0)

                preds_denorm = preds.detach().cpu().numpy() * label_std + label_mean
                labels_denorm = labels.detach().cpu().numpy() * label_std + label_mean
                batch_mae = float((np.abs(preds_denorm - labels_denorm)).mean())
                batch_mse = float(((preds_denorm - labels_denorm) ** 2).mean())
                running_mae += batch_mae * imgs.size(0)
                running_mse += batch_mse * imgs.size(0)
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mae': f'{batch_mae:.3f}'})

                preds_all.extend(preds_denorm.flatten().tolist())
                # for non-hybrid models, classification preds are same as rounded regression preds
                preds_cls_all.extend(list(np.rint(preds_denorm.flatten()).astype(int)))
                labels_all.extend(labels_denorm.flatten().tolist())

    val_loss = running_loss / len(loader.dataset)
    val_mae = running_mae / len(loader.dataset)
    val_rmse = math.sqrt(running_mse / len(loader.dataset))
    # safety: ensure aligned lengths for downstream metrics
    try:
        min_len = min(len(preds_all), len(labels_all), len(preds_cls_all))
        if len(preds_all) != min_len:
            preds_all = preds_all[:min_len]
        if len(labels_all) != min_len:
            labels_all = labels_all[:min_len]
        if len(preds_cls_all) != min_len:
            preds_cls_all = preds_cls_all[:min_len]
    except Exception:
        pass
    return val_loss, val_mae, val_rmse, preds_all, labels_all, preds_cls_all


def compute_accuracy_within(preds, labels, tol=5.0):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    if len(labels) == 0:
        return 0.0
    within = (np.abs(preds - labels) <= tol).sum()
    return float(within) / len(labels) * 100.0


def compute_r2(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    if len(labels) == 0:
        return 0.0
    ss_res = ((labels - preds) ** 2).sum()
    ss_tot = ((labels - labels.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


def run_training(model, train_loader, val_loader, criterion, optimizer, scheduler, ema, swa_model, swa_scheduler, args, device, label_mean, label_std, out_dir_suffix='', start_epoch=1):
    """Run the epoch loop and return best checkpoint path and final metrics."""
    best_val_mae = float('inf')
    best_ckpt = None
    use_amp = (device.type == 'cuda')

    for epoch in range(start_epoch, args.epochs + 1):
        print(f'Epoch {epoch}/{args.epochs} — train samples: {len(train_loader.dataset)} val samples: {len(val_loader.dataset)}')
        t0 = time.time()
        train_loss, train_mae = train_epoch(model, train_loader, optimizer, criterion, device, label_mean, label_std, use_amp=use_amp, mixup_alpha=args.mixup_alpha, cutmix_prob=args.cutmix_prob, cutmix_alpha=args.cutmix_alpha, accumulate_steps=max(1, args.accumulate_steps))
        # optional gradient clipping
        if args.clip_grad and args.clip_grad > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        # validate with EMA weights if requested
        if ema is not None:
            val_loss, val_mae, val_rmse, preds_all, labels_all, preds_cls_all = validate_epoch(ema.ema, val_loader, criterion, device, label_mean, label_std, tta=args.tta)
        else:
            val_loss, val_mae, val_rmse, preds_all, labels_all, preds_cls_all = validate_epoch(model, val_loader, criterion, device, label_mean, label_std, tta=args.tta)
        dt = time.time() - t0

        acc_within_1 = compute_accuracy_within(preds_all, labels_all, tol=1.0)
        acc_within_3 = compute_accuracy_within(preds_all, labels_all, tol=3.0)
        acc_within_5 = compute_accuracy_within(preds_all, labels_all, tol=5.0)
        try:
            preds_arr = np.asarray(preds_all)
            labels_arr = np.asarray(labels_all)
            preds_cls_arr = np.asarray(preds_cls_all)
            # exact-match: rounded prediction equals rounded label (integer equality)
            if len(labels_arr) > 0:
                preds_int = np.rint(preds_arr).astype(int)
                labels_int = np.rint(labels_arr).astype(int)
                overall_acc = float((preds_int == labels_int).sum()) / len(labels_arr) * 100.0
                # classification exact (from bins)
                class_exact = float((preds_cls_arr == labels_int).sum()) / len(labels_arr) * 100.0
            else:
                overall_acc = 0.0
                class_exact = 0.0
        except Exception:
            overall_acc = 0.0
            class_exact = 0.0
        r2 = compute_r2(preds_all, labels_all)
        print(f'  time: {dt:.1f}s  train_loss: {train_loss:.4f}  train_mae: {train_mae:.3f}  val_loss: {val_loss:.4f}  val_mae: {val_mae:.3f}  val_rmse: {val_rmse:.3f}  acc±1: {acc_within_1:.1f}%  acc±3: {acc_within_3:.1f}%  acc±5: {acc_within_5:.1f}%  exact: {overall_acc:.1f}%  class_exact: {class_exact:.1f}%  R2: {r2:.3f}')

        # scheduler step
        try:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        except Exception:
            pass

        # SWA step
        if swa_model is not None:
            try:
                swa_model.update_parameters(model)
                if swa_scheduler is not None:
                    swa_scheduler.step()
            except Exception:
                pass

        ckpt = Path(args.out) / f'{out_dir_suffix}epoch_{epoch}.pth'
        to_save = {'epoch': epoch, 'model_state': model.state_dict(), 'val_loss': val_loss, 'val_mae': val_mae, 'overall_exact': overall_acc}
        try:
            to_save['optimizer'] = optimizer.state_dict()
        except Exception:
            pass
        try:
            to_save['scheduler'] = scheduler.state_dict()
        except Exception:
            pass
        try:
            if ema is not None:
                to_save['ema_state'] = ema.state_dict()
        except Exception:
            pass
        try:
            if swa_model is not None:
                to_save['swa_model_state'] = swa_model.state_dict()
        except Exception:
            pass
        try:
            if swa_scheduler is not None:
                to_save['swa_scheduler'] = swa_scheduler.state_dict()
        except Exception:
            pass
        torch.save(to_save, ckpt)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_ckpt = Path(args.out) / f'{out_dir_suffix}best_model.pth'
            to_save_best = {'epoch': epoch, 'model_state': model.state_dict(), 'val_loss': val_loss, 'val_mae': val_mae, 'overall_exact': overall_acc}
            try:
                to_save_best['optimizer'] = optimizer.state_dict()
            except Exception:
                pass
            try:
                to_save_best['scheduler'] = scheduler.state_dict()
            except Exception:
                pass
            try:
                if ema is not None:
                    to_save_best['ema_state'] = ema.state_dict()
            except Exception:
                pass
            try:
                if swa_model is not None:
                    to_save_best['swa_model_state'] = swa_model.state_dict()
            except Exception:
                pass
            try:
                if swa_scheduler is not None:
                    to_save_best['swa_scheduler'] = swa_scheduler.state_dict()
            except Exception:
                pass
            torch.save(to_save_best, best_ckpt)
            print(f'  Saved best model to {best_ckpt} (val_mae: {best_val_mae:.3f}, exact: {overall_acc:.1f}%)')

        try:
            print(f'  Accuracy — ±1: {acc_within_1:.1f}%, ±3: {acc_within_3:.1f}%, ±5: {acc_within_5:.1f}%, exact: {overall_acc:.1f}%, class_exact: {class_exact:.1f}%')
        except Exception:
            pass

    print('Training complete for this run')
    try:
        preds_arr = np.asarray(preds_all)
        labels_arr = np.asarray(labels_all)
        if len(labels_arr) > 0:
            preds_int = np.rint(preds_arr).astype(int)
            labels_int = np.rint(labels_arr).astype(int)
            overall_acc = float((preds_int == labels_int).sum()) / len(labels_arr) * 100.0
        else:
            overall_acc = 0.0
    except Exception:
        overall_acc = 0.0
    print(f'Final Accuracy — ±1: {acc_within_1:.1f}%, ±3: {acc_within_3:.1f}%, ±5: {acc_within_5:.1f}%, overall_exact: {overall_acc:.1f}%')
    return best_ckpt, {'val_mae': best_val_mae, 'acc±1': acc_within_1, 'acc±3': acc_within_3, 'acc±5': acc_within_5, 'overall': overall_acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='out_all_plays')
    parser.add_argument('--labels', type=str, default='data/labels.csv')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', type=str, default='checkpoints')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--small', action='store_true', help='use mobilenet_v3_small for faster runs')
    parser.add_argument('--val-split', type=float, default=0.3)
    parser.add_argument('--use-timm', action='store_true', help='use timm model if available')
    parser.add_argument('--timm-backbone', type=str, default='tf_efficientnet_b4', help='backbone model for timm')
    parser.add_argument('--hybrid', action='store_true', help='use hybrid classification+regression head')
    parser.add_argument('--hybrid-bins', type=int, default=65, help='number of classification bins for hybrid head')
    parser.add_argument('--hybrid-min-bin', type=int, default=-32, help='minimum integer bin for hybrid head')
    parser.add_argument('--hybrid-class-weight', type=float, default=2.0, help='weight for classification loss when using hybrid head')
    parser.add_argument('--snap-threshold', type=float, default=0.6, help='confidence threshold to snap regression to classification bin during validation')
    parser.add_argument('--focal-alpha', type=float, default=0.0, help='alpha scaling for focal loss (0 to disable)')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='gamma parameter for focal loss (0 to disable)')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--mixup-alpha', type=float, default=0.2)
    parser.add_argument('--warmup-epochs', type=int, default=3)
    parser.add_argument('--img-size', type=int, default=224, help='input image size')
    parser.add_argument('--tta', type=int, default=0, help='simple TTA (0 or 1 for horizontal flip)')
    parser.add_argument('--use-ema', action='store_true', help='track EMA of model weights')
    parser.add_argument('--ema-decay', type=float, default=0.9999, help='EMA decay')
    parser.add_argument('--randaugment', action='store_true', help='use RandAugment for stronger augmentation')
    parser.add_argument('--cutmix-prob', type=float, default=0.0, help='probability to apply CutMix per batch')
    parser.add_argument('--cutmix-alpha', type=float, default=1.0, help='alpha parameter for CutMix beta distribution')
    parser.add_argument('--accumulate-steps', type=int, default=1, help='gradient accumulation steps to simulate larger batch size on CPU')
    parser.add_argument('--use-albumentations', action='store_true', help='use Albumentations for augmentation (optional dependency)')
    parser.add_argument('--freeze-epochs', type=int, default=0, help='number of initial epochs to freeze backbone and train head only')
    parser.add_argument('--kfold', action='store_true', help='run k-fold cross validation and save per-fold checkpoints')
    parser.add_argument('--folds', type=int, default=5, help='number of folds for k-fold training')
    parser.add_argument('--fine-tune-epochs', type=int, default=0, help='additional fine-tune epochs to run after each fold')
    parser.add_argument('--fine-tune-lr', type=float, default=1e-5, help='learning rate to use for fine-tuning')
    parser.add_argument('--onecycle', action='store_true', help='use OneCycleLR scheduler')
    parser.add_argument('--max-lr', type=float, default=1e-3, help='max LR for OneCycle')
    parser.add_argument('--swa', action='store_true', help='use SWA (stochastic weight averaging)')
    parser.add_argument('--swa-lr', type=float, default=1e-4, help='SWA LR')
    parser.add_argument('--clip-grad', type=float, default=0.0, help='max norm for gradient clipping (0 = disabled)')
    parser.add_argument('--persistent-workers', action='store_true', help='use persistent workers for DataLoader')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='DataLoader prefetch_factor (workers>0)')
    parser.add_argument('--torch-compile', action='store_true', help='compile the model with torch.compile (PyTorch 2.x)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from (loads model weights and starts from saved epoch+1)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    device = torch.device(args.device)

    # speedups when CUDA available
    if device.type == 'cuda':
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    aug_list = []
    if args.randaugment:
        try:
            aug_list.append(T.RandAugment())
        except Exception:
            # older torchvision versions may not have RandAugment
            pass
    # stronger CPU-friendly augmentations
    try:
        aug_list.append(T.AutoAugment())
    except Exception:
        pass
    aug_list.extend([
        T.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
    ])
    aug_list.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        # RandomErasing after ToTensor/Normalize (some torchvision versions accept it as a transform)
        aug_list.append(T.RandomErasing(p=0.2))
    except Exception:
        pass
    transform = T.Compose(aug_list)
    # if albumentations requested and available, build an albumentations pipeline and wrapper
    if args.use_albumentations and _HAS_ALB:
        try:
            alb_aug = A.Compose([
                A.RandomResizedCrop(args.img_size, args.img_size, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

            def alb_wrapper(img):
                # img: PIL Image -> numpy
                arr = np.array(img)
                res = alb_aug(image=arr)
                return res['image']

            transform = alb_wrapper
        except Exception:
            # fallback to torchvision pipeline
            pass
    elif args.use_albumentations and not _HAS_ALB:
        print('Albumentations requested but not installed; falling back to torchvision transforms')

    dataset = PlayImageDataset(images_dir=args.images, labels_csv=args.labels, transform=transform)
    n = len(dataset)
    if n == 0:
        raise SystemExit(f'No images found in {args.images} matching labels {args.labels}')

    val_size = int(n * args.val_split)
    train_size = n - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    try:
        train_indices = train_ds.indices
    except Exception:
        train_indices = list(range(train_size))

    train_labels = np.array([float(dataset.df.iloc[i]['yards_gained']) for i in train_indices])
    label_mean = float(train_labels.mean())
    label_std = float(train_labels.std()) if train_labels.std() > 0 else 1.0
    dataset.label_mean = label_mean
    dataset.label_std = label_std

    pin_memory = True if device.type == 'cuda' else False
    dl_kwargs = dict(batch_size=args.batch_size, pin_memory=pin_memory)
    # only set worker-related args if workers > 0
    if args.workers and args.workers > 0:
        dl_kwargs.update(dict(num_workers=args.workers, persistent_workers=bool(args.persistent_workers), prefetch_factor=args.prefetch_factor))
    else:
        dl_kwargs.update(dict(num_workers=0))

    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    # for validation use same workers settings but no shuffle
    val_dl_kwargs = dict(batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
    if args.workers and args.workers > 0:
        val_dl_kwargs.update(dict(num_workers=args.workers, persistent_workers=bool(args.persistent_workers), prefetch_factor=args.prefetch_factor))
    else:
        val_dl_kwargs.update(dict(num_workers=0))
    val_loader = DataLoader(val_ds, **val_dl_kwargs)

    model = make_model(pretrained=args.pretrained, device=device, small=args.small,
                       use_timm=args.use_timm, timm_backbone=args.timm_backbone)

    # Hybrid classification+regression head: CLI flags will trigger wrapping the backbone
    if args.hybrid:
        # only supported reliably for non-timm backbones (mobilenet variants)
        if args.use_timm:
            warnings.warn('Hybrid head with timm backbones is experimental; proceeding but may fail')
        # determine feature dim from backbone's classifier first linear layer where possible
        feat_dim = None
        try:
            # for torchvision MobileNet classifier[0] is Linear
            feat_dim = model.classifier[0].in_features
        except Exception:
            # fallback to try attribute num_features
            feat_dim = getattr(model, 'num_features', None)
        if feat_dim is None:
            # last resort: run a dummy forward to infer feature size
            with torch.no_grad():
                dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
                out = model(dummy)
                feat_dim = out.view(1, -1).shape[1]
        # Replace classifier so backbone returns features for HybridModel
        # We'll set backbone.classifier to identity mapping to return the intermediate features
        try:
            # if classifier is Sequential ending with Linear->1, replace it with a simple projector to feat_dim
            model.classifier = nn.Sequential(nn.Identity())
        except Exception:
            pass
        model = HybridModel(model, feat_dim, n_bins=args.hybrid_bins, min_bin=args.hybrid_min_bin).to(device)
        # attach the classification weight to the wrapped model so train/val loops can access it
        try:
            model.class_weight = float(args.hybrid_class_weight)
        except Exception:
            model.class_weight = 1.0
        # attach focal loss params and snapping threshold
        try:
            model.focal_alpha = float(args.focal_alpha)
            model.focal_gamma = float(args.focal_gamma)
            model.snap_threshold = float(args.snap_threshold)
        except Exception:
            model.focal_alpha = 0.0
            model.focal_gamma = 0.0
            model.snap_threshold = 0.0

    start_epoch = 1
    loaded_ckpt = None
    # resume from checkpoint if provided
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            print(f'Loading checkpoint {ckpt_path} ...')
            try:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            except TypeError:
                # older torch versions do not support weights_only kwarg
                ckpt = torch.load(ckpt_path, map_location=device)
            loaded_ckpt = ckpt
            # support different checkpoint key names
            state = None
            if isinstance(ckpt, dict):
                for key in ('model_state', 'state_dict', 'model_state_dict', 'model'):
                    if key in ckpt:
                        state = ckpt[key]
                        break
            if state is None:
                state = ckpt
            try:
                model.load_state_dict(state)
            except Exception:
                new_state = {}
                for k, v in state.items():
                    new_key = k
                    for prefix in ('module.', 'model.', 'encoder.'):
                        if new_key.startswith(prefix):
                            new_key = new_key[len(prefix):]
                            break
                    new_state[new_key] = v
                model.load_state_dict(new_state)
            if isinstance(ckpt, dict) and 'epoch' in ckpt:
                start_epoch = int(ckpt['epoch']) + 1
            print(f'Resuming from epoch {start_epoch}')
        else:
            print(f'Warning: resume checkpoint {ckpt_path} not found, starting from epoch 1')

    # optional torch.compile for PyTorch 2.x to speed up training (if available)
    if args.torch_compile:
        try:
            model = torch.compile(model)
        except Exception:
            pass

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    try:
        base_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    except Exception:
        base_scheduler = optim.lr.scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Optionally use OneCycleLR
    if args.onecycle:
        try:
            total_steps = max(1, args.epochs * max(1, len(train_loader)))
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, total_steps=total_steps)
            use_base_scheduler = False
        except Exception:
            scheduler = base_scheduler
            use_base_scheduler = False

    if args.warmup_epochs and args.warmup_epochs > 0:
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                return float(epoch + 1) / float(max(1, args.warmup_epochs))
            return 1.0
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        use_base_scheduler = True
    else:
        scheduler = base_scheduler
        use_base_scheduler = False

    # helper to attempt loading optimizer/scheduler/ema/swa states from a checkpoint dict
    def try_load_states(ckpt_dict, opt, sched, ema_obj, swa_m, swa_s):
        if not ckpt_dict or not isinstance(ckpt_dict, dict):
            return
        # optimizer
        if opt is not None and 'optimizer' in ckpt_dict:
            try:
                opt.load_state_dict(ckpt_dict['optimizer'])
                print('Loaded optimizer state from checkpoint')
            except Exception:
                print('Warning: failed to load optimizer state (shape mismatch or missing params)')
        # scheduler
        if sched is not None and 'scheduler' in ckpt_dict:
            try:
                sched.load_state_dict(ckpt_dict['scheduler'])
                print('Loaded scheduler state from checkpoint')
            except Exception:
                print('Warning: failed to load scheduler state')
        # EMA
        if ema_obj is not None and 'ema_state' in ckpt_dict:
            try:
                # ema_obj is ModelEMA, has attribute `ema` (nn.Module)
                ema_obj.ema.load_state_dict(ckpt_dict['ema_state'])
                print('Loaded EMA state from checkpoint')
            except Exception:
                print('Warning: failed to load EMA state')
        # SWA
        if swa_m is not None and 'swa_model_state' in ckpt_dict:
            try:
                swa_m.load_state_dict(ckpt_dict['swa_model_state'])
                print('Loaded SWA averaged model state from checkpoint')
            except Exception:
                print('Warning: failed to load SWA model state')
        if swa_s is not None and 'swa_scheduler' in ckpt_dict:
            try:
                swa_s.load_state_dict(ckpt_dict['swa_scheduler'])
                print('Loaded SWA scheduler state from checkpoint')
            except Exception:
                print('Warning: failed to load SWA scheduler state')

    best_val_mae = float('inf')
    use_amp = (device.type == 'cuda')

    ema = None
    if args.use_ema:
        ema = ModelEMA(model, decay=args.ema_decay, device=device)
        # attach to optimizer for easy updates in train loop
        setattr(optimizer, '_ema', ema)
    # attempt to load optimizer/scheduler/ema/swa now that ema exists
    # (actual restore happens after SWA objects are created below)

    # Optional SWA
    swa_model = None
    swa_scheduler = None
    if args.swa:
        try:
            from torch.optim import swa_utils
            swa_model = swa_utils.AveragedModel(model)
            swa_scheduler = swa_utils.SWALR(optimizer, swa_lr=args.swa_lr)
        except Exception:
            swa_model = None
            swa_scheduler = None

    # If K-fold requested, orchestrate folds
    if args.kfold and args.folds > 1:
        print(f'Running {args.folds}-fold cross validation')
        indices = np.arange(n)
        rng = np.random.RandomState(42)
        rng.shuffle(indices)
        folds = np.array_split(indices, args.folds)
        fold_metrics = []
        for fi in range(args.folds):
            val_idx = folds[fi]
            train_idx = np.concatenate([f for i, f in enumerate(folds) if i != fi])
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            train_dl = DataLoader(train_subset, shuffle=True, **dl_kwargs)
            val_dl = DataLoader(val_subset, **val_dl_kwargs)

            # fresh model and optimizer for each fold
            m = make_model(pretrained=args.pretrained, device=device, small=args.small, use_timm=args.use_timm, timm_backbone=args.timm_backbone)
            if args.torch_compile:
                try:
                    m = torch.compile(m)
                except Exception:
                    pass
            opt = optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            try:
                base_sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
            except Exception:
                base_sched = optim.lr.scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)
            sched = base_sched
            ema_f = ModelEMA(m, decay=args.ema_decay, device=device) if args.use_ema else None
            if ema_f is not None:
                setattr(opt, '_ema', ema_f)
            swa_m = None
            swa_s = None
            if args.swa:
                try:
                    from torch.optim import swa_utils
                    swa_m = swa_utils.AveragedModel(m)
                    swa_s = swa_utils.SWALR(opt, swa_lr=args.swa_lr)
                except Exception:
                    swa_m = None
                    swa_s = None

            out_suffix = f'fold{fi}_'
            best_ckpt, metrics = run_training(m, train_dl, val_dl, criterion, opt, sched, ema_f, swa_m, swa_s, args, device, label_mean, label_std, out_dir_suffix=out_suffix)
            fold_metrics.append(metrics)
            # show exact-match accuracy explicitly for this fold
            print(f'Fold {fi} exact-match accuracy: {metrics.get("overall", 0.0):.2f}%')

            # optional fine-tuning stage using the best checkpoint as initialization
            if args.fine_tune_epochs and args.fine_tune_epochs > 0 and best_ckpt is not None:
                print(f'Fine-tuning fold {fi} model for {args.fine_tune_epochs} more epochs (lr={args.fine_tune_lr})')
                # load best weights
                state = torch.load(best_ckpt, map_location=device)
                try:
                    m.load_state_dict(state['model_state'])
                except Exception:
                    # try partial load
                    m_state = m.state_dict()
                    for k, v in state.get('model_state', {}).items():
                        if k in m_state and m_state[k].shape == v.shape:
                            m_state[k] = v
                    m.load_state_dict(m_state)
                # rebuild optimizer and scheduler for fine-tuning
                opt_ft = optim.AdamW(m.parameters(), lr=args.fine_tune_lr, weight_decay=args.weight_decay)
                try:
                    sched_ft = optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=max(1, args.fine_tune_epochs))
                except Exception:
                    sched_ft = optim.lr.scheduler.ReduceLROnPlateau(opt_ft, mode='min', factor=0.5, patience=2)
                ema_ft = ModelEMA(m, decay=args.ema_decay, device=device) if args.use_ema else None
                if ema_ft is not None:
                    setattr(opt_ft, '_ema', ema_ft)
                swa_m_ft = None
                swa_s_ft = None
                if args.swa:
                    try:
                        from torch.optim import swa_utils
                        swa_m_ft = swa_utils.AveragedModel(m)
                        swa_s_ft = swa_utils.SWALR(opt_ft, swa_lr=args.swa_lr)
                    except Exception:
                        swa_m_ft = None
                        swa_s_ft = None
                args_ft = argparse.Namespace(**vars(args))
                args_ft.epochs = args.fine_tune_epochs
                # run fine-tuning on the same train/val splits
                run_training(m, train_dl, val_dl, criterion, opt_ft, sched_ft, ema_ft, swa_m_ft, swa_s_ft, args_ft, device, label_mean, label_std, out_dir_suffix=f'fold{fi}_finetune_')
        # summarize folds
        print('K-fold results:')
        for i, fm in enumerate(fold_metrics):
            print(f' Fold {i}: val_mae={fm.get("val_mae"):.4f} acc±1={fm.get("acc±1"):.1f}% acc±3={fm.get("acc±3"):.1f}% acc±5={fm.get("acc±5"):.1f}% overall_exact={fm.get("overall", 0.0):.2f}%')
        return

    def freeze_backbone(m):
        # freeze everything except the classifier/head we added
        for name, p in m.named_parameters():
            p.requires_grad = False
        # try common classifier attributes
        for attr in ('classifier', 'fc', 'head'):
            if hasattr(m, attr):
                try:
                    mod = getattr(m, attr)
                    for p in mod.parameters():
                        p.requires_grad = True
                    return
                except Exception:
                    continue
        # fallback: unfreeze last two parameters
        params = list(m.parameters())
        if len(params) >= 2:
            params[-1].requires_grad = True
            params[-2].requires_grad = True

    def unfreeze_all(m):
        for p in m.parameters():
            p.requires_grad = True

    # initially freeze backbone if requested
    if args.freeze_epochs and args.freeze_epochs > 0:
        freeze_backbone(model)
        # recreate optimizer to include only trainable params (head)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        # try to restore optimizer/scheduler/ema/swa states for the recreated optimizer
        try_load_states(loaded_ckpt, optimizer, scheduler, ema, swa_model, swa_scheduler)
        # rebuild scheduler consistent with new optimizer
        try:
            base_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
        except Exception:
            base_scheduler = optim.lr.scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        if args.onecycle:
            try:
                total_steps = max(1, args.epochs * max(1, len(train_loader)))
                scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, total_steps=total_steps)
                use_base_scheduler = False
            except Exception:
                scheduler = base_scheduler
                use_base_scheduler = False
        else:
            scheduler = base_scheduler
            use_base_scheduler = False
        # reattach ema if used
        if args.use_ema:
            setattr(optimizer, '_ema', ema)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f'Epoch {epoch}/{args.epochs} — train samples: {len(train_ds)} val samples: {len(val_ds)}')
        t0 = time.time()
        # if we're finishing the freeze period, unfreeze and rebuild optimizer/scheduler
        if args.freeze_epochs and epoch == (args.freeze_epochs + 1):
            unfreeze_all(model)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            try:
                base_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
            except Exception:
                base_scheduler = optim.lr.scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
            if args.onecycle:
                try:
                    total_steps = max(1, args.epochs * max(1, len(train_loader)))
                    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, total_steps=total_steps)
                    use_base_scheduler = False
                except Exception:
                    scheduler = base_scheduler
                    use_base_scheduler = False
            else:
                scheduler = base_scheduler
                use_base_scheduler = False
            if args.use_ema:
                setattr(optimizer, '_ema', ema)
            # try to reload optimizer/scheduler/ema/swa states after unfreezing and rebuilding optimizer
            try_load_states(loaded_ckpt, optimizer, scheduler, ema, swa_model, swa_scheduler)

        train_loss, train_mae = train_epoch(model, train_loader, optimizer, criterion, device, label_mean, label_std, use_amp=use_amp, mixup_alpha=args.mixup_alpha, cutmix_prob=args.cutmix_prob, cutmix_alpha=args.cutmix_alpha, accumulate_steps=max(1, args.accumulate_steps))
        # optional gradient clipping
        if args.clip_grad and args.clip_grad > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        # validate with EMA weights if requested
        if ema is not None:
            # temporarily swap into ema. We'll evaluate ema.ema directly.
            val_loss, val_mae, val_rmse, preds_all, labels_all, preds_cls_all = validate_epoch(ema.ema, val_loader, criterion, device, label_mean, label_std, tta=args.tta)
        else:
            val_loss, val_mae, val_rmse, preds_all, labels_all, preds_cls_all = validate_epoch(model, val_loader, criterion, device, label_mean, label_std, tta=args.tta)
        dt = time.time() - t0

        acc_within_1 = compute_accuracy_within(preds_all, labels_all, tol=1.0)
        acc_within_3 = compute_accuracy_within(preds_all, labels_all, tol=3.0)
        acc_within_5 = compute_accuracy_within(preds_all, labels_all, tol=5.0)
        # overall accuracy: exact-match after rounding predictions and labels
        try:
            preds_arr = np.asarray(preds_all)
            labels_arr = np.asarray(labels_all)
            preds_cls_arr = np.asarray(preds_cls_all)
            if len(labels_arr) > 0:
                preds_int = np.rint(preds_arr).astype(int)
                labels_int = np.rint(labels_arr).astype(int)
                overall_acc = float((preds_int == labels_int).sum()) / len(labels_arr) * 100.0
                class_exact = float((preds_cls_arr == labels_int).sum()) / len(labels_arr) * 100.0
            else:
                overall_acc = 0.0
                class_exact = 0.0
        except Exception:
            overall_acc = 0.0
            class_exact = 0.0
        r2 = compute_r2(preds_all, labels_all)
        print(f'  time: {dt:.1f}s  train_loss: {train_loss:.4f}  train_mae: {train_mae:.3f}  val_loss: {val_loss:.4f}  val_mae: {val_mae:.3f}  val_rmse: {val_rmse:.3f}  acc±1: {acc_within_1:.1f}%  acc±3: {acc_within_3:.1f}%  acc±5: {acc_within_5:.1f}%  exact: {overall_acc:.1f}%  R2: {r2:.3f}')

        try:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
                if use_base_scheduler and hasattr(base_scheduler, 'step'):
                    base_scheduler.step()

            # SWA step and update averaged weights
            if swa_model is not None:
                try:
                    swa_model.update_parameters(model)
                    if swa_scheduler is not None:
                        swa_scheduler.step()
                except Exception:
                    pass
        except Exception:
            pass

        ckpt = Path(args.out) / f'epoch_{epoch}.pth'
        to_save = {'epoch': epoch, 'model_state': model.state_dict(), 'val_loss': val_loss, 'val_mae': val_mae, 'overall_exact': overall_acc}
        try:
            to_save['optimizer'] = optimizer.state_dict()
        except Exception:
            pass
        try:
            to_save['scheduler'] = scheduler.state_dict()
        except Exception:
            pass
        try:
            if ema is not None:
                to_save['ema_state'] = ema.state_dict()
        except Exception:
            pass
        try:
            if swa_model is not None:
                to_save['swa_model_state'] = swa_model.state_dict()
        except Exception:
            pass
        try:
            if swa_scheduler is not None:
                to_save['swa_scheduler'] = swa_scheduler.state_dict()
        except Exception:
            pass
        torch.save(to_save, ckpt)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_ckpt = Path(args.out) / 'best_model.pth'
            to_save_best = {'epoch': epoch, 'model_state': model.state_dict(), 'val_loss': val_loss, 'val_mae': val_mae, 'overall_exact': overall_acc}
            try:
                to_save_best['optimizer'] = optimizer.state_dict()
            except Exception:
                pass
            try:
                to_save_best['scheduler'] = scheduler.state_dict()
            except Exception:
                pass
            try:
                if ema is not None:
                    to_save_best['ema_state'] = ema.state_dict()
            except Exception:
                pass
            try:
                if swa_model is not None:
                    to_save_best['swa_model_state'] = swa_model.state_dict()
            except Exception:
                pass
            try:
                if swa_scheduler is not None:
                    to_save_best['swa_scheduler'] = swa_scheduler.state_dict()
            except Exception:
                pass
            torch.save(to_save_best, best_ckpt)
            print(f'  Saved best model to {best_ckpt} (val_mae: {best_val_mae:.3f}, exact: {overall_acc:.1f}%)')

        try:
            print(f'  Accuracy — ±1: {acc_within_1:.1f}%, ±3: {acc_within_3:.1f}%, ±5: {acc_within_5:.1f}%, exact: {overall_acc:.1f}%')
        except Exception:
            pass

    print('Training complete')
    try:
        # overall accuracy: exact-match after rounding predictions and labels for last epoch
        try:
            preds_arr = np.asarray(preds_all)
            labels_arr = np.asarray(labels_all)
            if len(labels_arr) > 0:
                preds_int = np.rint(preds_arr).astype(int)
                labels_int = np.rint(labels_arr).astype(int)
                overall_acc = float((preds_int == labels_int).sum()) / len(labels_arr) * 100.0
            else:
                overall_acc = 0.0
        except Exception:
            overall_acc = 0.0
        print(f'Final Accuracy — ±1: {acc_within_1:.1f}%, ±3: {acc_within_3:.1f}%, ±5: {acc_within_5:.1f}%, overall_exact: {overall_acc:.1f}%')
    except Exception:
        pass


if __name__ == '__main__':
    main()
