# Adversarial Robustness of Deep Learning-Based Thyroid Nodule Segmentation in Ultrasound

**Authors:** Nicholas Dietrich, David McShannon

**Corresponding Author:** Nicholas Dietrich, Temerty Faculty of Medicine, University of Toronto (nicholas.dietrich@mail.utoronto.ca)

## Overview

This repository provides implementations of two black-box adversarial attacks and three inference-time defenses for ultrasound segmentation, as described in the paper.

**Attacks:**
- Structured Speckle Amplification Attack (SSAA) — boundary-targeted multiplicative speckle noise
- Frequency-Domain Ultrasound Attack (FDUA) — Butterworth bandpass-filtered phase perturbations

**Defenses:**
- Randomized Preprocessing with Test-Time Augmentation
- Deterministic Input Denoising
- Stochastic Ensemble with Consistency-Aware Aggregation

## Dataset

Experiments use the [Stanford AIMI Thyroid Ultrasound Cine-clip dataset](https://doi.org/10.71718/7m5n-rh16) (192 nodules, 167 patients). Download the dataset and update paths in your scripts accordingly.

## Installation

```bash
pip install -r requirements.txt
```

## Repository Structure

```
├── model.py        # U-Net segmentation architecture
├── train.py        # Model training script
├── attacks.py      # SSAA and FDUA attack implementations
├── defenses.py     # Three inference-time defense strategies
├── metrics.py      # Evaluation metrics (Dice, IoU, SSIM, HD95)
├── config.py       # Hyperparameters
└── README.md
```

## Reproducing Results

### Step 1: Train the segmentation model

```bash
python train.py --data_path /path/to/dataset.hdf5 --split_path /path/to/data_split.json
```

This trains a U-Net following the architecture and training protocol described in the paper. The best checkpoint is saved as `unet_best.pth`.

### Step 2: Run attacks and defenses

```python
import numpy as np
import torch
from model import UNet
from attacks import ssaa_attack, fdua_attack
from defenses import predict_with_defense
from metrics import predict, dice

# Load your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(features=(32, 64, 128, 256), drop=0.0).to(device)
ckpt = torch.load("unet_best.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Load a frame and mask from the dataset
frame = ...   # float32 numpy array, shape (256, 256), range [0, 1]
mask_gt = ... # binary numpy array, shape (256, 256)

# Run SSAA attack
adv_image, attack_results = ssaa_attack(frame, mask_gt, model, device)
print(f"SSAA Dice drop: {attack_results['dice_drop']:.3f}")

# Evaluate defense on attacked image
defended_pred = predict_with_defense(model, adv_image, device, defense="defense2")
print(f"Defended Dice: {dice(defended_pred, mask_gt):.3f}")
```

## Attack Parameters

| Parameter | SSAA | FDUA |
|-----------|------|------|
| Query budget | 500 (50 iter × 10 pop) | 500 (50 iter × 10 pop) |
| Amplitude | 0.03–0.20 | 0.05–0.50 (epsilon) |
| Spatial extent | sigma 3–30 px, offset 0–15 px | Low: 5–38 cycles, High: 38–102 cycles |
| Noise model | Rayleigh (multiplicative) | Phase noise [-π, π] (multiplicative) |

## Citation

```bibtex
@article{dietrich2026adversarial,
  title={Adversarial Robustness of Deep Learning-Based Thyroid Nodule Segmentation in Ultrasound},
  author={Dietrich, Nicholas and McShannon, David},
  journal={},
  year={2026}
}
```

## License

This code is released under the MIT License. See [LICENSE](LICENSE) for details.
