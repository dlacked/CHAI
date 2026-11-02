"""
Fast variant of train_classify.py: reads pre-cropped tooth JPEGs from
ResNet/cropped_tooth/{split}/{jaw}/{stem}_{fdi_number}.jpg instead of decoding the full arch
photo and cropping on every __getitem__ call.

The precrop cache was built with the same CROP_PAD=10 convention train_classify.py's
GhorbaniToothDataset uses (int-truncated box), just via PIL's float-coordinate .crop() instead of
cv2 array slicing - a sub-pixel difference that's negligible for training a brand-new model from
scratch (see ResNet/fast_dataset.py's docstring for the same reasoning, already applied to
CHAI's own ablation training this session). The cache's 671,615 train-split crops match this
script's own "~670k train tooth instances" count, confirming full coverage of Ghorbani's tooth
population (same GT JSON, same jaws, same digit range).

Motivation: train_classify.py's on-the-fly, single-threaded (num_workers=0, forced by a Windows
spawn-multiprocessing MemoryError) cropping made one run take >24h without finishing its fixed 40
epochs. This script removes the per-tooth full-image decode+crop entirely, so each __getitem__ is
just one small-JPEG read.

Usage:
    .venv/Scripts/python.exe comparison/ghorbani/train_classify_fast.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

GHORBANI_DIR = Path(__file__).resolve().parent
if str(GHORBANI_DIR) not in sys.path:
    sys.path.append(str(GHORBANI_DIR))
from train_classify import (  # noqa: E402
    FDI_CLASSES, FDI_TO_IDX, IMAGE_EXTS, IMG_SIZE, DATASET_DIR, MODEL_DIR, RUNS_DIR,
    build_model, run_epoch,
)

from config import PathConfig
PROJECT_ROOT = PathConfig.ROOT
CROP_CACHE_DIR = PROJECT_ROOT / "ResNet" / "cropped_tooth"


class GhorbaniToothDatasetFast(Dataset):
    """Drop-in replacement for train_classify.py's GhorbaniToothDataset, reading from the
    precrop cache. Same __getitem__ contract: (img_tensor, class_idx)."""

    def __init__(self, dataset_dir, split, transform=None):
        self.split = split
        self.transform = transform
        self.items = []  # (jaw, image_name, fdi_number, class_idx)

        for jaw in ("lower", "upper"):
            json_dir = Path(dataset_dir) / split / "labels_json" / jaw
            if not json_dir.exists():
                continue
            for json_path in sorted(json_dir.glob("*.json")):
                stem = json_path.stem
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for t in data.get("tooth", []):
                    fdi_number = t.get("teeth_num")
                    seg = t.get("segmentation", [])
                    if fdi_number not in FDI_TO_IDX or not seg:
                        continue
                    crop_path = CROP_CACHE_DIR / split / jaw / f"{stem}_{fdi_number}.jpg"
                    if not crop_path.exists():
                        continue
                    self.items.append((jaw, stem, fdi_number, FDI_TO_IDX[fdi_number]))

        print(f"{split}: {len(self.items)} tooth crops indexed from precrop cache "
              f"({len(FDI_CLASSES)} classes).")

    def reshuffle(self):
        pass  # DataLoader(shuffle=True) handles this directly now - no per-image cache to protect.

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        jaw, stem, fdi_number, class_idx = self.items[idx]
        crop_path = CROP_CACHE_DIR / self.split / jaw / f"{stem}_{fdi_number}.jpg"
        image = Image.open(crop_path).convert("RGB")
        img_tensor = self.transform(image) if self.transform else transforms.ToTensor()(image)
        return img_tensor, class_idx


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = GhorbaniToothDatasetFast(DATASET_DIR, "train", transform=train_transform)
    val_ds = GhorbaniToothDatasetFast(DATASET_DIR, "val", transform=eval_transform)
    # shuffle=True is safe now (no per-image cache locality to protect, unlike the on-the-fly version).
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    epochs = 40  # per Ghorbani et al.'s stated Stage 2 setup - fixed epoch count, no early stop

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "train_f1": [], "val_f1": []}

    for epoch in range(epochs):
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, device, criterion, optimizer, tag="Train"
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, device, criterion, optimizer=None, tag="Val"
        )

        for key, val in zip(history, (train_loss, val_loss, train_acc, val_acc, train_f1, val_f1)):
            history[key].append(val)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "classify_best.pth")
            print(f"  --> Saved new best model (Val Acc: {best_acc:.4f})")

    with open(RUNS_DIR / "classify_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining completed. Best Validation Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
