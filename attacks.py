"""
Adversarial attacks for ultrasound segmentation.

Structured Speckle Amplification Attack (SSAA):
    Exploits the multiplicative speckle noise model by injecting spatially
    structured Rayleigh-distributed noise concentrated near the predicted
    segmentation boundary.

Frequency-Domain Ultrasound Attack (FDUA):
    Introduces perturbations by modifying the 2D Fourier transform within a
    targeted frequency band using a Butterworth bandpass filter and random
    phase noise.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt

from metrics import predict, dice, imperceptibility


def ssaa_attack(frame, mask_gt, model, device, n_iter=50, pop=10):
    """
    Structured Speckle Amplification Attack.

    Args:
        frame: Input image, float32 [0,1], shape (H, W).
        mask_gt: Ground truth binary mask, shape (H, W).
        model: Segmentation model.
        device: Torch device.
        n_iter: Number of search iterations.
        pop: Population size per iteration.

    Returns:
        best_adv: Adversarial image, float32 [0,1].
        results: Dict with attack metrics.
    """
    clean_pred = predict(model, frame, device)
    clean_d = dice(clean_pred, mask_gt)

    bdist = (
        distance_transform_edt(1 - clean_pred)
        if clean_pred.sum() > 0
        else np.ones_like(frame) * 50
    )

    best_adv, best_d, best_p = frame.copy(), clean_d, {}
    q = 0

    for _ in range(n_iter):
        for j in range(pop):
            alpha = np.random.uniform(0.03, 0.20)
            sigma = np.random.uniform(3, 30)
            offset = np.random.uniform(0, 15)

            G = np.exp(-0.5 * ((bdist - offset) / max(sigma, 1e-6)) ** 2)
            noise = np.random.rayleigh(1.0, frame.shape).astype(np.float32)
            noise = gaussian_filter(noise, 1.5)
            noise = (noise - noise.mean()) / (noise.std() + 1e-6)

            adv = np.clip(frame * (1 + alpha * G * noise), 0, 1).astype(
                np.float32
            )
            ad = dice(predict(model, adv, device), mask_gt)
            q += 1

            if ad < best_d:
                best_d, best_adv = ad, adv.copy()
                best_p = {"alpha": alpha, "sigma": sigma, "offset": offset}

    imp = imperceptibility(frame, best_adv)
    return best_adv, {
        "attack": "SSAA",
        "clean_dice": clean_d,
        "adv_dice": best_d,
        "dice_drop": clean_d - best_d,
        **imp,
        "queries": q,
        "params": best_p,
    }


def butterworth_bp(shape, low, high, order=2):
    """Butterworth bandpass filter in the frequency domain."""
    r, c = shape
    y, x = np.ogrid[-r // 2 : r - r // 2, -c // 2 : c - c // 2]
    d = np.sqrt(x * x + y * y).astype(np.float32)
    d[d == 0] = 1e-6
    hp = (
        1.0 / (1.0 + (low / d) ** (2 * order))
        if low > 0
        else np.ones(shape, np.float32)
    )
    lp = (
        1.0 / (1.0 + (d / high) ** (2 * order))
        if high > 0
        else np.ones(shape, np.float32)
    )
    return (hp * lp).astype(np.float32)


def fdua_attack(frame, mask_gt, model, device, n_iter=50, pop=10):
    """
    Frequency-Domain Ultrasound Attack.

    Args:
        frame: Input image, float32 [0,1], shape (H, W).
        mask_gt: Ground truth binary mask, shape (H, W).
        model: Segmentation model.
        device: Torch device.
        n_iter: Number of search iterations.
        pop: Population size per iteration.

    Returns:
        best_adv: Adversarial image, float32 [0,1].
        results: Dict with attack metrics.
    """
    clean_pred = predict(model, frame, device)
    clean_d = dice(clean_pred, mask_gt)

    F = np.fft.fftshift(np.fft.fft2(frame))
    mf = min(frame.shape) // 2  # Nyquist frequency

    best_adv, best_d, best_p = frame.copy(), clean_d, {}
    q = 0

    for _ in range(n_iter):
        for j in range(pop):
            eps = np.random.uniform(0.05, 0.5)
            low = np.random.uniform(5, mf * 0.3)
            high = np.random.uniform(mf * 0.3, mf * 0.8)
            high = max(high, low + 5)
            order = np.random.choice([1, 2, 3, 4])

            H = butterworth_bp(frame.shape, low, high, order)
            noise_ph = np.random.uniform(-np.pi, np.pi, frame.shape).astype(
                np.float32
            )
            F_adv = F * (1 + eps * H * np.exp(1j * noise_ph))
            adv = np.clip(
                np.real(np.fft.ifft2(np.fft.ifftshift(F_adv))), 0, 1
            ).astype(np.float32)

            ad = dice(predict(model, adv, device), mask_gt)
            q += 1

            if ad < best_d:
                best_d, best_adv = ad, adv.copy()
                best_p = {
                    "eps": eps,
                    "low": low,
                    "high": high,
                    "order": int(order),
                }

    imp = imperceptibility(frame, best_adv)
    return best_adv, {
        "attack": "FDUA",
        "clean_dice": clean_d,
        "adv_dice": best_d,
        "dice_drop": clean_d - best_d,
        **imp,
        "queries": q,
        "params": best_p,
    }
