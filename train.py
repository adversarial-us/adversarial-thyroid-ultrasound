"""
Train U-Net for thyroid nodule segmentation.

Follows the training protocol described in the paper:
- AdamW optimizer (lr=1e-4, weight_decay=1e-5)
- Combined binary cross-entropy and Dice loss
- Cosine annealing warm restarts scheduler (T0=20, T_mult=2)
- Batch size 16
- Early stopping with patience 15
- Augmentations: random horizontal/vertical flips, random 90-degree rotations
- Images resized to 256x256, normalized to [0, 1]

Usage:
    python train.py --data_path /path/to/dataset.hdf5 --split_path /path/to/data_split.json
"""

import argparse
import json
import os
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from skimage.transform import resize

from model import UNet


class ThyroidDataset(Dataset):
    """Dataset loader for Stanford AIMI Thyroid Ultrasound Cine-clip."""

    def __init__(self, hdf5_path, indices, image_size=256, augment=False):
        self.hdf5_path = hdf5_path
        self.indices = indices
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        with h5py.File(self.hdf5_path, "r") as hdf5:
            frame = hdf5["image"][idx].astype(np.float32) / 255.0
            mask = (hdf5["mask"][idx] > 0).astype(np.float32)

        frame = resize(
            frame, (self.image_size, self.image_size),
            anti_aliasing=True, preserve_range=True
        ).astype(np.float32)
        mask = resize(
            mask, (self.image_size, self.image_size),
            order=0, preserve_range=True
        ).astype(np.float32)

        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                frame = np.flip(frame, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            # Random vertical flip
            if np.random.rand() > 0.5:
                frame = np.flip(frame, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
            # Random 90-degree rotation
            k = np.random.randint(0, 4)
            frame = np.rot90(frame, k).copy()
            mask = np.rot90(mask, k).copy()

        frame = torch.from_numpy(frame).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        return frame, mask


class DiceBCELoss(nn.Module):
    """Combined binary cross-entropy and Dice loss."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1.0
        dice_loss = 1.0 - (2.0 * (probs * targets).sum() + smooth) / (
            probs.sum() + targets.sum() + smooth
        )
        return bce_loss + dice_loss


def dice_score(pred, target):
    """Compute Dice coefficient from binary tensors."""
    smooth = 1.0
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return float(
        (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    )


def build_data_splits(hdf5_path, split_path):
    """Build train/val/test index lists from data split JSON."""
    with open(split_path) as f:
        split_info = json.load(f)

    with h5py.File(hdf5_path, "r") as hdf5:
        all_aids = [x.decode().strip() for x in hdf5["annot_id"][:]]

    train_nodules = set(split_info["train_nodules"])
    val_nodules = set(split_info["val_nodules"])
    test_nodules = set(split_info["test_nodules"])

    train_idx, val_idx, test_idx = [], [], []
    for i, aid in enumerate(all_aids):
        if aid in train_nodules:
            train_idx.append(i)
        elif aid in val_nodules:
            val_idx.append(i)
        elif aid in test_nodules:
            test_idx.append(i)

    return train_idx, val_idx, test_idx


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_idx, val_idx, _ = build_data_splits(args.data_path, args.split_path)
    print(f"Train: {len(train_idx)} frames, Val: {len(val_idx)} frames")

    train_ds = ThyroidDataset(args.data_path, train_idx, augment=True)
    val_ds = ThyroidDataset(args.data_path, val_idx, augment=False)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

    # Model
    model = UNet(features=(32, 64, 128, 256), drop=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2
    )
    loss_fn = DiceBCELoss()

    # Training loop
    best_val_dice = 0.0
    patience_counter = 0
    patience = 15
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.max_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for frames, masks in train_dl:
            frames, masks = frames.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(frames)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_dl)

        # Validate
        model.eval()
        val_dices = []
        with torch.no_grad():
            for frames, masks in val_dl:
                frames, masks = frames.to(device), masks.to(device)
                preds = (torch.sigmoid(model(frames)) > 0.5).float()
                for j in range(preds.shape[0]):
                    val_dices.append(dice_score(preds[j], masks[j]))
        val_dice = np.mean(val_dices)

        print(
            f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Early stopping
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            save_path = os.path.join(args.output_dir, "unet_best.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_dice": val_dice,
                },
                save_path,
            )
            print(f"  -> Saved best model (Val Dice: {val_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete. Best Val Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for thyroid segmentation")
    parser.add_argument("--data_path", required=True, help="Path to dataset.hdf5")
    parser.add_argument("--split_path", required=True, help="Path to data_split.json")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory")
    parser.add_argument("--max_epochs", type=int, default=200, help="Maximum epochs")
    args = parser.parse_args()
    train(args)
