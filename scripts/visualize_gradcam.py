#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# Import training model builder
from train_cnn import HybridModel, make_model


def build_eval_transform(img_size=224):
    # Deterministic eval transform aligned with training normalization
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(fwd_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            try:
                h.remove()
            except Exception:
                pass

    def generate(self, input_tensor, target_scalar):
        # Forward
        self.model.zero_grad()
        out = self.model(input_tensor)
        if isinstance(out, tuple):
            reg, logits = out
            y = target_scalar(reg, logits)
        else:
            y = target_scalar(out, None)

        # Backward from scalar output
        y.backward(retain_graph=False)

        # Grad-CAM: global-average-pool gradients, weight activations, ReLU, then upsample
        grads = self.gradients  # (B, C, H, W)
        acts = self.activations # (B, C, H, W)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)
        # Normalize to [0,1]
        cam_min = cam.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        cam_max = cam.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam


def overlay_heatmap(img, heatmap):
    # img: PIL Image (RGB), heatmap: numpy (H,W) in [0,1]
    import cv2
    img_np = np.array(img)
    hmap = (heatmap * 255).astype(np.uint8)
    hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    hmap = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB)
    # Alpha blend
    overlay = (0.4 * hmap + 0.6 * img_np).astype(np.uint8)
    return Image.fromarray(overlay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--images', type=str, required=True)
    ap.add_argument('--out', type=str, default='gradcam_out')
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--head', type=str, choices=['regression', 'classification'], default='regression')
    ap.add_argument('--limit', type=int, default=50)
    ap.add_argument('--small', action='store_true')
    ap.add_argument('--pretrained', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    # Build model matching training backbone
    model = make_model(pretrained=args.pretrained, device=device, small=args.small, use_timm=False)
    # Wrap as HybridModel like training when hybrid used; infer feat dim
    try:
        feat_dim = model.classifier[0].in_features
    except Exception:
        feat_dim = 1280
    backbone = model
    # Return features from backbone
    backbone.classifier = torch.nn.Sequential(torch.nn.Identity())
    model = HybridModel(backbone, feat_dim, n_bins=65, min_bin=-32).to(device)

    # Load checkpoint safely
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt.get('model_state', ckpt)
    model.load_state_dict(state)
    model.eval()

    # Target layer: last conv block of MobileNetV3-small
    target_layer = model.backbone.features[-1]
    cam = GradCAM(model, target_layer)

    transform = build_eval_transform(args.img_size)

    # Helper to select target scalar
    def target_scalar(reg, logits):
        if args.head == 'regression':
            # Single sample scalar
            return reg.view(-1)[0]
        else:
            # Top-1 logit to avoid softmax artifacts
            top_idx = torch.argmax(logits, dim=1)
            return logits[0, top_idx.item()]

    # Collect images
    img_paths = []
    for p in Path(args.images).glob('**/*.png'):
        img_paths.append(str(p))
        if len(img_paths) >= args.limit:
            break
    for p in Path(args.images).glob('**/*.jpg'):
        img_paths.append(str(p))
        if len(img_paths) >= args.limit:
            break

    for i, ip in enumerate(img_paths):
        try:
            pil = Image.open(ip).convert('RGB')
        except Exception:
            continue
        inp = transform(pil).unsqueeze(0).to(device)

        with torch.enable_grad():
            heat = cam.generate(inp, target_scalar)

        # Upsample CAM to input spatial size
        heat_up = F.interpolate(heat, size=(pil.size[1], pil.size[0]), mode='bilinear', align_corners=False)
        heat_np = heat_up.squeeze().detach().cpu().numpy()

        overlay = overlay_heatmap(pil, heat_np)
        out_path = Path(args.out) / f'cam_{args.head}_{i:04d}.png'
        overlay.save(out_path)

    cam.remove_hooks()
    print(f'Saved {len(img_paths)} Grad-CAM overlays to {args.out}')


if __name__ == '__main__':
    main()
