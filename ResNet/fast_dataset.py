"""
Fast CHAI tooth dataset for TRAINING (not evaluation): reads pre-cropped tooth JPEGs from
ResNet/cropped_tooth/{split}/{jaw}/{stem}_{fdi}.jpg instead of decoding the full arch photo
and cropping on every __getitem__ call, the way train_chai.py's ToothDatasetChai does.

Why this is safe for training but was rejected for evaluation (see evaluate_chai.py's docstring):
the precrop cache was built with PIL's float-coordinate .crop(), which differs by a sub-pixel
amount from the int()-truncated pad=10 box ToothDatasetChai/the published Table 1/2 numbers use -
that tiny difference matters for reproducing an exact, already-published, McNemar-tested accuracy
figure, but is completely negligible for training a NEW model from scratch (training already
applies random color-jitter augmentation on top, so a sub-pixel crop boundary shift is far smaller
than the augmentation noise the model is trained to be robust to anyway).

Motivation: train_chai_no_pretrain.py's first run (2026-09-23) sat for 1.5+ hours without
completing a single epoch or writing a single checkpoint - confirmed via checkpoint.pth's absence,
not just quiet stdout - while GPU memory was allocated but utilization read ~0%, consistent with
the training loop being almost entirely bottlenecked on ToothDatasetChai's per-tooth full-image
cv2.imread calls (num_workers=0, no persistent per-tooth crop cache, only crop *coordinates* are
cached) rather than actual GPU compute. This dataset class exists specifically to remove that
bottleneck for ablation training runs that don't need pixel-exact reproduction of a specific
already-published number.

Usage: import ToothDatasetChaiFast and use it as a drop-in replacement for ToothDatasetChai
(same __getitem__ signature: img_tensor, meta, target_label).
"""
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

TOOTH_DIR = Path(__file__).resolve().parent
CROP_CACHE_DIR = TOOTH_DIR / "cropped_tooth"


class ToothDatasetChaiFast(Dataset):
    """Drop-in replacement for train_chai.py's ToothDatasetChai, reading from the precrop cache."""

    def __init__(self, csv_path, dataset_dir, split, transform=None):
        self.rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                class12 = row.get("class12")
                if class12 is not None and class12 != "" and 0 <= int(class12) <= 11:
                    self.rows.append(row)
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_name = row["image_name"]
        jaw = row["jaw"]
        fdi_number = row["fdi_number"]
        target_label = int(row["class12"])

        stem = Path(image_name).stem
        crop_path = CROP_CACHE_DIR / self.split / jaw / f"{stem}_{fdi_number}.jpg"
        image = Image.open(crop_path).convert("RGB")

        if self.transform:
            img_tensor = self.transform(image)
        else:
            from torchvision import transforms
            img_tensor = transforms.ToTensor()(image)

        meta = torch.tensor([
            float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]),
        ], dtype=torch.float32)

        return img_tensor, meta, target_label
