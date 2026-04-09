"""
Evaluation metrics for segmentation quality and adversarial imperceptibility.
"""

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import distance_transform_edt


def predict(model, frame, device):
    """Run model inference on a single 2D frame. Returns binary mask."""
    with torch.no_grad():
        t = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(device)
        return (torch.sigmoid(model(t)) > 0.5).float().cpu().numpy()[0, 0]


def predict_proba(model, frame, device):
    """Run model inference, returning probability map (not thresholded)."""
    with torch.no_grad():
        t = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(device)
        return torch.sigmoid(model(t)).cpu().numpy()[0, 0]


def dice(a, b):
    """Dice similarity coefficient with Laplace smoothing."""
    return float((2 * (a * b).sum() + 1) / (a.sum() + b.sum() + 1))


def iou(a, b):
    """Intersection over Union with Laplace smoothing."""
    inter = (a * b).sum()
    return float((inter + 1) / (a.sum() + b.sum() - inter + 1))


def imperceptibility(clean, adv):
    """Compute imperceptibility metrics between clean and adversarial images."""
    d = adv - clean
    return {
        "l2": float(np.sqrt((d**2).sum())),
        "linf": float(np.abs(d).max()),
        "ssim": float(ssim(clean, adv, data_range=1.0)),
    }


def hausdorff_95(p, g):
    """95th percentile Hausdorff distance between binary masks."""
    if p.sum() == 0 or g.sum() == 0:
        return float("nan")
    pb = p & ~(distance_transform_edt(p) > 1)
    gb = g & ~(distance_transform_edt(g) > 1)
    if pb.sum() == 0 or gb.sum() == 0:
        return float("nan")
    dt_g = distance_transform_edt(~gb)
    dt_p = distance_transform_edt(~pb)
    return float(max(np.percentile(dt_g[pb], 95), np.percentile(dt_p[gb], 95)))
