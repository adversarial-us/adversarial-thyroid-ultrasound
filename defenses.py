"""
Inference-time defenses for adversarial robustness.

Defense 1 - Randomized Preprocessing with Test-Time Augmentation:
    Applies K=5 random spatial rescalings and Gaussian blurs, averages
    probability maps, and thresholds at 0.5.

Defense 2 - Deterministic Input Denoising:
    Applies Gaussian blur (sigma=1.0) followed by 3x3 median filter
    before standard inference. Single forward pass.

Defense 3 - Stochastic Ensemble with Consistency-Aware Aggregation:
    Generates K=5 augmented copies with random shifts, rescaling, blur,
    noise, and brightness. Weights predictions by pixel-wise agreement
    with majority vote.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.ndimage import shift as ndi_shift
from skimage.transform import resize

from metrics import predict, predict_proba
from config import CFG


# ── Defense 1: Randomized Preprocessing with TTA ──


def random_preprocess(frame, resize_range=(0.9, 1.1), blur_range=(0.3, 1.5)):
    """Apply random spatial rescaling and Gaussian blur."""
    h, w = frame.shape
    scale = np.random.uniform(*resize_range)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = resize(
        frame, (new_h, new_w), anti_aliasing=True, preserve_range=True
    ).astype(np.float32)
    out = np.zeros((h, w), dtype=np.float32)
    y_off = (new_h - h) // 2
    x_off = (new_w - w) // 2
    if scale >= 1.0:
        out = resized[y_off : y_off + h, x_off : x_off + w]
    else:
        py, px = -y_off, -x_off
        out[py : py + new_h, px : px + new_w] = resized
    sigma = np.random.uniform(*blur_range)
    out = gaussian_filter(out, sigma=sigma)
    return np.clip(out, 0, 1).astype(np.float32)


def predict_defense1(model, frame, device, K=5):
    """Randomized preprocessing with test-time augmentation."""
    proba_sum = np.zeros(frame.shape, dtype=np.float64)
    for _ in range(K):
        aug = random_preprocess(frame, CFG.tta_resize_range, CFG.tta_blur_range)
        proba_sum += predict_proba(model, aug, device)
    return (proba_sum / K > 0.5).astype(np.float32)


# ── Defense 2: Deterministic Input Denoising ──


def denoise_frame(frame, sigma=1.0, med_k=3):
    """Apply Gaussian blur followed by median filter."""
    out = gaussian_filter(frame, sigma=sigma)
    out = median_filter(out, size=med_k)
    return np.clip(out, 0, 1).astype(np.float32)


def predict_defense2(model, frame, device):
    """Deterministic input denoising defense."""
    denoised = denoise_frame(
        frame, sigma=CFG.denoise_sigma, med_k=CFG.denoise_median
    )
    return predict(model, denoised, device)


# ── Defense 3: Stochastic Ensemble with Consistency-Aware Aggregation ──


def diverse_augmentation(frame):
    """Generate a single diverse augmentation of the input frame."""
    h, w = frame.shape
    aug = frame.copy()

    # Random spatial shift
    dy, dx = np.random.uniform(-4, 4), np.random.uniform(-4, 4)
    aug = ndi_shift(aug, [dy, dx], order=1, mode="reflect")

    # Random rescaling
    scale = np.random.uniform(0.93, 1.07)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = resize(
        aug, (new_h, new_w), anti_aliasing=True, preserve_range=True
    ).astype(np.float32)
    out = np.zeros((h, w), dtype=np.float32)
    y_off = (new_h - h) // 2
    x_off = (new_w - w) // 2
    if scale >= 1.0:
        out = resized[y_off : y_off + h, x_off : x_off + w]
    else:
        py, px = -y_off, -x_off
        out[py : py + new_h, px : px + new_w] = resized
    aug = out

    # Random Gaussian blur
    aug = gaussian_filter(aug, sigma=np.random.uniform(0.2, 1.2))

    # Random additive Gaussian noise
    aug = aug + np.random.normal(
        0, np.random.uniform(0.005, 0.02), aug.shape
    ).astype(np.float32)

    # Random brightness shift
    aug = aug + np.random.uniform(-0.03, 0.03)

    return np.clip(aug, 0, 1).astype(np.float32)


def predict_defense4(model, frame, device, K=5):
    """
    Stochastic ensemble with consistency-aware aggregation.

    Returns:
        pred: Binary prediction mask.
        consistency: Per-pixel agreement map across K predictions.
    """
    probas = np.array(
        [
            predict_proba(model, diverse_augmentation(frame), device)
            for _ in range(K)
        ]
    )
    votes = (probas > 0.5).astype(np.float32)
    majority = (votes.mean(axis=0) > 0.5).astype(np.float32)

    agreement = np.array(
        [(votes[k] == majority).astype(np.float32) for k in range(K)]
    )
    consistency = agreement.sum(axis=0) / K

    weighted = np.zeros(frame.shape, dtype=np.float64)
    wsum = np.zeros(frame.shape, dtype=np.float64)
    for k in range(K):
        weighted += probas[k] * agreement[k]
        wsum += agreement[k]
    final_proba = weighted / np.maximum(wsum, 1e-8)
    pred = (final_proba > 0.5).astype(np.float32)
    return pred, consistency


# ── Unified interface ──


def predict_with_defense(model, frame, device, defense="none", K=5):
    """
    Run inference with optional defense.

    Args:
        defense: One of "none", "defense1", "defense2", "defense4".
    """
    if defense == "none":
        return predict(model, frame, device)
    elif defense == "defense1":
        return predict_defense1(model, frame, device, K)
    elif defense == "defense2":
        return predict_defense2(model, frame, device)
    elif defense == "defense4":
        pred, _ = predict_defense4(model, frame, device, K)
        return pred
    else:
        raise ValueError(f"Unknown defense: {defense}")
