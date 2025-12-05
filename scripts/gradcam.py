#!/usr/bin/env python3
"""
Create Grad-CAM heatmaps for images using a trained model checkpoint.

Usage example:
  python3 scripts/gradcam.py --checkpoint checkpoints_highacc_run/best_model.pth \
      --images out_all_plays --out gradcam_results --img-size 224 --use-timm --timm-backbone efficientnet_b0

The script is CPU-friendly by default but accepts `--device` to use CUDA if available.
"""
import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import timm


def build_model(use_timm=False, timm_backbone='efficientnet_b0', small=False, pretrained=False, img_size=224):
    if use_timm:
        model = timm.create_model(timm_backbone, pretrained=pretrained, num_classes=1)
        return model
    else:
        # fallback to a small MobileNetV3-like model using torchvision
        from torchvision.models import mobilenet_v3_small, mobilenet_v3_large

        if small:
            m = mobilenet_v3_small(pretrained=pretrained)
        else:
            m = mobilenet_v3_large(pretrained=pretrained)
        # replace classifier with single output
        in_features = m.classifier[0].in_features if hasattr(m.classifier[0], 'in_features') else m.classifier[-1].in_features
        m.classifier = torch.nn.Sequential(torch.nn.Linear(in_features, 1))
        return m


class HybridModel(torch.nn.Module):
    def __init__(self, backbone, feat_dim, n_bins=65, min_bin=-32):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.n_bins = int(n_bins)
        self.min_bin = int(min_bin)
        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 1)
        )
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, self.n_bins)
        )

    def forward(self, x):
        feats = self.backbone(x)
        if feats.dim() > 2:
            feats = feats.view(feats.size(0), -1)
        reg = self.reg_head(feats)
        logits = self.cls_head(feats)
        return reg, logits


def build_hybrid_backbone(small=True, pretrained=True, img_size=224):
    m = models.mobilenet_v3_small(pretrained=pretrained) if small else models.mobilenet_v3_large(pretrained=pretrained)
    # set classifier to identity to expose features
    m.classifier = torch.nn.Sequential(torch.nn.Identity())
    # infer feature dimension by dummy forward
    with torch.no_grad():
        dummy = torch.randn(1, 3, img_size, img_size)
        out = m(dummy)
        feat_dim = out.view(1, -1).shape[1]
    return m, int(feat_dim)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def save_grad(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        def save_activation(module, input, output):
            self.activations = output.detach()

        target_layer.register_forward_hook(save_activation)
        target_layer.register_backward_hook(save_grad)

    def __call__(self, input_tensor, class_idx=None):
        # forward
        output = self.model(input_tensor)
        # Handle hybrid models that return (regression, logits)
        if isinstance(output, tuple):
            reg_preds, logits = output
            score = reg_preds.squeeze()
        else:
            if output.dim() == 2 and output.size(1) == 1:
                score = output.squeeze(1).mean() if class_idx is None else output[0, class_idx]
            else:
                # regression: use scalar output
                score = output.squeeze()

        self.model.zero_grad()
        if score.requires_grad:
            score.backward(retain_graph=True)
        grads = self.gradients  # [B, C, H, W]
        acts = self.activations  # [B, C, H, W]
        # global-average-pool gradients
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)
        # normalize per-sample
        cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
        cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
        denom = (cam_max - cam_min).clamp(min=1e-6)
        cam_norm = (cam - cam_min) / denom
        return cam_norm.squeeze(1).cpu().numpy()  # [B, H, W]


def overlay_heatmap_on_image(img: Image.Image, heatmap: np.ndarray, colormap='jet', alpha=0.5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm

    heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype('uint8')).resize(img.size, resample=Image.BILINEAR)) / 255.0
    cmap = cm.get_cmap(colormap)
    colored = cmap(heatmap_resized)[:, :, :3]
    img_np = np.array(img).astype(np.float32) / 255.0
    overlay = (1 - alpha) * img_np + alpha * colored
    overlay = (overlay * 255).astype('uint8')
    return Image.fromarray(overlay)


def find_target_layer(model):
    # Heuristics: for timm EfficientNet, use conv_head; for mobilenet use features[-1]
    for name, module in reversed(list(model.named_modules())):
        # pick a 4D activation conv layer
        if isinstance(module, torch.nn.Conv2d):
            return module
    # fallback: use last module
    return list(model.modules())[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--images', default='out_all_plays', help='Folder with images')
    parser.add_argument('--out', default='gradcam_results', help='Output folder to save heatmaps')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--use-timm', action='store_true')
    parser.add_argument('--timm-backbone', default='efficientnet_b0')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--ext', default='.jpg', help='Image extension filter (e.g. .png, .jpg)')
    parser.add_argument('--threshold', type=float, default=None, help='If set, copy/move images with max heatmap >= threshold to high-importance folder (0-1)')
    parser.add_argument('--high-out', default=None, help='Folder to place high-importance images (if --threshold set).')
    parser.add_argument('--move', action='store_true', help='Move high-importance originals instead of copying')
    parser.add_argument('--save-overlay', action='store_true', help='Save overlay images (default: True)')
    parser.add_argument('--copy-originals', action='store_true', help='Also copy originals to <out>/originals (default: True)')
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

    # Decide model architecture based on checkpoint contents
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = None
    if isinstance(checkpoint, dict):
        # Prefer EMA > SWA > model_state > state_dict
        for key in ('ema_state', 'swa_model_state', 'model_state', 'state_dict', 'model_state_dict'):
            if key in checkpoint:
                state = checkpoint[key]
                break
    if state is None:
        state = checkpoint

    # Heuristic: if keys include reg_head/cls_head, it's a HybridModel
    keys = list(state.keys()) if isinstance(state, dict) else []
    is_hybrid = any(k.startswith('reg_head.') for k in keys) or any(k.startswith('cls_head.') for k in keys)

    if is_hybrid:
        backbone, feat_dim = build_hybrid_backbone(small=True, pretrained=args.pretrained, img_size=args.img_size)
        model = HybridModel(backbone, feat_dim)
    else:
        model = build_model(use_timm=args.use_timm, timm_backbone=args.timm_backbone, small=args.small, pretrained=args.pretrained, img_size=args.img_size)

    # load checkpoint
    # PyTorch 2.6 defaults weights_only=True which can fail for older checkpoints
    # Use weights_only=False for trusted local checkpoints
    # load weights with key adaptation
    try:
        model.load_state_dict(state)
    except Exception:
        new_state = {}
        for k, v in state.items():
            new_key = k.replace('module.', '')
            if new_key.startswith('model.'):
                new_key = new_key[len('model.'):]
            if new_key.startswith('encoder.'):
                new_key = new_key[len('encoder.'):]
            new_state[new_key] = v
        model.load_state_dict(new_state)

    model.to(device)

    target_layer = find_target_layer(model)
    cam = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # also store originals for reference
    orig_dir = out_dir / 'originals'
    if args.copy_originals:
        orig_dir.mkdir(parents=True, exist_ok=True)

    high_out_dir = None
    if args.threshold is not None:
        if args.high_out is None:
            high_out_dir = out_dir / 'high_importance'
        else:
            high_out_dir = Path(args.high_out)
        high_out_dir.mkdir(parents=True, exist_ok=True)

    exts = [args.ext.lower()]
    files = [p for p in img_dir.rglob('*') if p.suffix.lower() in exts]
    if not files:
        # try common extensions
        files = [p for p in img_dir.rglob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]

    for p in files:
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            continue
        inp = transform(img).unsqueeze(0).to(device)
        # Grad-CAM requires gradients -- do forward/backward with grad enabled
        inp.requires_grad = True
        heat = cam(inp)
        heatmap = heat[0]
        # resize heatmap to original image size
        heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype('uint8')).resize(img.size, resample=Image.BILINEAR)) / 255.0
        # save overlay if requested
        out_path = out_dir / p.name
        if args.save_overlay:
            overlay = overlay_heatmap_on_image(img, heatmap_resized, alpha=0.5)
            overlay.save(out_path)
        # copy or move originals
        if args.copy_originals:
            img.save(orig_dir / p.name)

        # threshold filtering
        if args.threshold is not None:
            max_activation = float(heatmap.max())
            if max_activation >= args.threshold:
                target = high_out_dir / p.name
                if args.move:
                    try:
                        p.replace(target)
                    except Exception:
                        # fallback to copy
                        img.save(target)
                else:
                    img.save(target)

    print(f"Saved Grad-CAM overlays to {out_dir}")


if __name__ == '__main__':
    main()
