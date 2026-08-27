"""
Baseline comparison attack: SimBA (Simple Black-box Adversarial Attacks).

Adapted from Guo et al. (ICML 2019) for segmentation. Uses the pixel
basis with a fixed step size. The attack objective is adapted for
segmentation by minimizing the Dice similarity coefficient.

Reference:
    Guo C, Gardner JR, You Y, Wilson AG, Weinberger KQ.
    Simple Black-box Adversarial Attacks.
    Proceedings of the 36th International Conference on Machine
    Learning (ICML) 2019;97:2484-93.
"""

import numpy as np

from metrics import predict, dice, imperceptibility


def simba_attack(frame, mask_gt, model, device,
                 n_queries=500, epsilon=0.05):
    """
    SimBA (pixel basis) adapted for segmentation.

    Iterates through randomly ordered pixel coordinates, perturbing
    each by a fixed step size. The positive perturbation is evaluated
    first; if it does not reduce DSC, the negative perturbation is
    evaluated. The change is retained only when it improves the attack
    objective. Each coordinate evaluation requires one or two queries.

    Args:
        frame: Input image, float32 [0,1], shape (H, W).
        mask_gt: Ground truth binary mask, shape (H, W).
        model: Segmentation model.
        device: Torch device.
        n_queries: Maximum query budget (default 500).
        epsilon: Perturbation step size per pixel (default 0.05).

    Returns:
        adv: Adversarial image, float32 [0,1].
        results: Dict with attack metrics.
    """
    h, w = frame.shape
    adv = frame.copy()
    clean_pred = predict(model, frame, device)
    clean_d = dice(clean_pred, mask_gt)
    best_d = clean_d
    queries = 0

    pixel_order = np.random.permutation(h * w)

    for pix_idx in pixel_order:
        if queries >= n_queries:
            break

        r, c = divmod(int(pix_idx), w)
        old_val = adv[r, c]

        # Try positive perturbation
        adv[r, c] = np.clip(old_val + epsilon, 0.0, 1.0)
        pred_plus = predict(model, adv, device)
        d_plus = dice(pred_plus, mask_gt)
        queries += 1

        if d_plus < best_d:
            best_d = d_plus
            continue

        if queries >= n_queries:
            adv[r, c] = old_val
            break

        # Try negative perturbation
        adv[r, c] = np.clip(old_val - epsilon, 0.0, 1.0)
        pred_minus = predict(model, adv, device)
        d_minus = dice(pred_minus, mask_gt)
        queries += 1

        if d_minus < best_d:
            best_d = d_minus
            continue

        # Neither improved; revert
        adv[r, c] = old_val

    imp = imperceptibility(frame, adv)
    return adv, {
        "attack": "SimBA",
        "clean_dice": clean_d,
        "adv_dice": best_d,
        "dice_drop": clean_d - best_d,
        **imp,
        "queries": queries,
        "epsilon": epsilon,
    }
