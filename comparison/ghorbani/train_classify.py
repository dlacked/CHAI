"""
Stage 2 of the Ghorbani et al. reproduction: an on-the-fly-cropped YOLOv8n-cls tooth classifier,
trained directly against CHAI's GT JSON + images - mirroring ResNet/tooth/train.py's ToothDataset
(crop happens in __getitem__, nothing is ever written to disk) instead of pre-cropping ~1M small
files to disk, which would have meaningfully dented the ~57GB free on this machine for no real
benefit over cropping in-memory like CHAI's own ResNet already does.

Classes are the full two-digit FDI number (24 classes: 11-16/21-26/31-36/41-46 - this dataset has
no 7/8 wisdom-tooth labels, see the paper's Limitations), pooled across both jaws - see
prepare_detect_labels.py's sibling design note for why (Ghorbani's own pipeline isn't described as
jaw-segregated at the model level).

Trains ultralytics' actual YOLOv8n-cls architecture (via ClassificationModel + ImageNet-pretrained
backbone, not the high-level ultralytics Trainer, which only accepts a folder-based dataset) with
a plain PyTorch loop - matching ResNet/tooth/train.py's own manual-loop pattern rather than
ultralytics' CLI trainer, since that's the only way to keep the on-the-fly crop and skip disk
writes entirely.

Hyperparameters match the paper's stated Stage 2 setup: 128x128 input, lr 1e-4, batch 64, 40
epochs.

Usage:
    .venv/Scripts/python.exe comparison/ghorbani/train_classify.py
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score

from ultralytics import YOLO
from ultralytics.nn.tasks import ClassificationModel

GHORBANI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GHORBANI_DIR.parent.parent
DATASET_DIR = PROJECT_ROOT.parent / "dataset"
MODEL_DIR = GHORBANI_DIR / "model"
RUNS_DIR = GHORBANI_DIR / "runs"

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".webp"]
CROP_PAD = 10  # matches ResNet/tooth/train.py's ToothDataset crop padding
IMG_SIZE = 128  # per Ghorbani et al.'s stated Stage 2 input size

# Full two-digit FDI numbers this dataset actually has GT for (digits 1-6 only, both jaws pooled -
# see this module's docstring for why 7/8 and per-jaw splitting are both out of scope here).
FDI_CLASSES = [q * 10 + d for q in (1, 2, 3, 4) for d in range(1, 7)]
FDI_TO_IDX = {n: i for i, n in enumerate(FDI_CLASSES)}


def poly_bbox(seg):
    if isinstance(seg[0], list) and len(seg[0]) == 2:
        poly = np.array(seg, dtype=float)
    else:
        poly = np.array(seg, dtype=float).reshape(-1, 2)
    return poly.min(axis=0), poly.max(axis=0)


class GhorbaniToothDataset(Dataset):
    """Indexes every (jaw, image, tooth) triple from the GT JSON at init time (cheap - just
    metadata), then crops the actual pixels lazily per __getitem__ call, exactly like
    ResNet/tooth/train.py's ToothDataset - nothing gets written to disk.

    Items are grouped by source image (each image contributes ~10-12 teeth) and reshuffle()
    shuffles at that group level rather than per-tooth, so __getitem__ calls (with
    DataLoader(shuffle=False) - see main()) stay on the same image for ~10-12 consecutive calls
    in a row. Combined with the single-image cache below, this cuts disk reads/decodes by
    roughly the same ~10-12x factor instead of re-reading+decoding a tooth's source image once
    per tooth in it - without this, workers=0 (needed to dodge Windows' spawn-multiprocessing
    MemoryError - see train_detect.py) made a single epoch over ~670k train tooth instances
    take hours instead of minutes."""

    def __init__(self, dataset_dir, split, transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.transform = transform
        self._groups = []  # list of per-image lists of (jaw, image_name, x1, y1, x2, y2, class_idx)
        self.items = []
        self._cache_key = None
        self._cache_image = None

        for jaw in ("lower", "upper"):
            json_dir = self.dataset_dir / split / "labels_json" / jaw
            image_dir = self.dataset_dir / split / "images" / jaw
            if not json_dir.exists():
                continue

            for json_path in sorted(json_dir.glob("*.json")):
                stem = json_path.stem
                image_name = None
                for ext in IMAGE_EXTS:
                    if (image_dir / f"{stem}{ext}").exists():
                        image_name = f"{stem}{ext}"
                        break
                if image_name is None:
                    continue

                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                group = []
                for t in data.get("tooth", []):
                    fdi_number = t.get("teeth_num")
                    seg = t.get("segmentation", [])
                    if fdi_number not in FDI_TO_IDX or not seg:
                        continue
                    (x_min, y_min), (x_max, y_max) = poly_bbox(seg)
                    group.append((jaw, image_name, x_min, y_min, x_max, y_max, FDI_TO_IDX[fdi_number]))
                if group:
                    self._groups.append(group)

        self.reshuffle()
        print(f"{split}: {len(self.items)} tooth crops indexed across {len(self._groups)} images "
              f"({len(FDI_CLASSES)} classes).")

    def reshuffle(self):
        """Shuffles source-image order (not per-tooth order) and flattens back into self.items -
        call once before each epoch instead of DataLoader's own shuffle=True, which operates at
        the per-tooth level and would defeat the single-image cache in __getitem__."""
        random.shuffle(self._groups)
        self.items = [item for group in self._groups for item in group]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        jaw, image_name, x_min, y_min, x_max, y_max, class_idx = self.items[idx]

        cache_key = (jaw, image_name)
        if cache_key == self._cache_key:
            image = self._cache_image
        else:
            img_path = self.dataset_dir / self.split / "images" / jaw / image_name
            image = cv2.imread(str(img_path))
            if image is None:
                image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            self._cache_key = cache_key
            self._cache_image = image
        h, w = image.shape[:2]

        x1 = int(max(0, x_min - CROP_PAD))
        y1 = int(max(0, y_min - CROP_PAD))
        x2 = int(min(w, x_max + CROP_PAD))
        y2 = int(min(h, y_max + CROP_PAD))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        img_tensor = self.transform(pil_img) if self.transform else transforms.ToTensor()(pil_img)
        return img_tensor, class_idx


def build_model(device):
    """Fresh YOLOv8n-cls architecture sized for FDI_CLASSES, backbone initialized from
    ultralytics' ImageNet-pretrained yolov8n-cls.pt (only the final head is randomly initialized,
    since its shape doesn't match the pretrained 1000-class head)."""
    model = ClassificationModel(cfg="yolov8n-cls.yaml", ch=3, nc=len(FDI_CLASSES), verbose=False)
    pretrained = YOLO("yolov8n-cls.pt")
    model.load(pretrained.model)
    return model.to(device)


def run_epoch(model, loader, device, criterion, optimizer=None, log_every=200, tag="Train"):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss, total_n = 0.0, 0
    all_preds, all_labels = [], []
    n_batches = len(loader)
    with torch.enable_grad() if train_mode else torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            if train_mode:
                optimizer.zero_grad()
            logits = model(images)
            # ultralytics' ClassificationModel.forward returns a plain Tensor in train() mode
            # but a (logits, logits) tuple in eval() mode (its own validator unpacks this same
            # way) - the val loop below runs under model.eval(), so without this it fed
            # CrossEntropyLoss a tuple instead of a Tensor and raised
            # "cross_entropy_loss(): argument 'input' (position 1) must be Tensor, not tuple".
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = criterion(logits, labels)
            if train_mode:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_n += labels.size(0)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            # Printed periodically rather than every batch (there are ~10k+ batches/epoch on the
            # full train set) so progress is visible without flooding the console - the on-the-fly
            # per-tooth cropping (see GhorbaniToothDataset) has no other progress signal, and its
            # per-batch cost is uneven enough that a silent run can look identical to a hang.
            if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == n_batches:
                running_acc = accuracy_score(all_labels, all_preds)
                print(f"  [{tag}] Batch [{batch_idx + 1}/{n_batches}] | Loss: {loss.item():.4f} | "
                      f"Running Acc: {running_acc:.4f}")

    avg_loss = total_loss / max(total_n, 1)
    acc = accuracy_score(all_labels, all_preds) if total_n else 0.0
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if total_n else 0.0
    return avg_loss, acc, f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # No horizontal flip - unlike generic image classification, flipping a tooth crop changes
    # which side of the midline it visually resembles, which would corrupt the FDI-number label.
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = GhorbaniToothDataset(DATASET_DIR, "train", transform=train_transform)
    val_ds = GhorbaniToothDataset(DATASET_DIR, "val", transform=eval_transform)
    # num_workers=0: Windows' "spawn" multiprocessing start method pickles the whole worker
    # process state (train_detect.py hit a MemoryError from exactly this) - GhorbaniToothDataset
    # itself is small, but not worth the same risk for the modest speedup on this machine.
    # shuffle=False on both: train_ds.reshuffle() (called once per epoch below) already
    # randomizes source-image order every epoch - a DataLoader-level shuffle would scramble
    # individual teeth again and defeat GhorbaniToothDataset's single-image cache, turning a
    # ~10-12x reduction in disk reads back into re-reading each tooth's source image from scratch.
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
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
        train_ds.reshuffle()
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
