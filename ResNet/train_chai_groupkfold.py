"""
GroupKFold variant of train_chai.py: trains the same ToothPositionClassifierNoTheta (pooled,
no-mirror, 12-way class12) model, but against one of GroupKFold's 10 folds (see
chai_paper_todo.md's CV-methodology plan, and finalize_group_kfold_split.py / the combined
functions/features/main_chai_groupkfold.py CSV) instead of the old fixed train/val split.

Reads the single combined ResNet/csv/features_groupkfold.csv (one row per tooth, every image in
train+val+test, tagged with cv_role="pool"/"test" and cv_fold=0..9) and, for a given --fold k:
  - train = rows where cv_role=="pool" and cv_fold != k
  - val   = rows where cv_role=="pool" and cv_fold == k
(cv_role=="test" rows are the fixed held-out test set - never used here, only by a separate eval
script run once per fold's trained model afterwards.)

Each row's own orig_split column (NOT a single Dataset-level split) is used to locate the image/
JSON on disk, since a pool row may physically live under the OLD train/, val/, or test/ directory
- nothing was physically moved, only the CV role changed.

train_chai.py and its outputs (ResNet/model/, ResNet/runs/) are left completely alone; this
writes to ResNet/model_cv/fold{k}/ and ResNet/runs_cv/fold{k}/.

Usage:
    .venv/Scripts/python.exe ResNet/train_chai_groupkfold.py --fold 0
    .venv/Scripts/python.exe ResNet/train_chai_groupkfold.py  # loops all 10 folds sequentially
"""
import sys
import argparse
import csv
import io
import time
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tooth_dir = Path(__file__).resolve().parent
if str(tooth_dir) not in sys.path:
    sys.path.append(str(tooth_dir))
from train_mirrored import ToothPositionClassifierNoTheta

_bootstrap_root = Path(__file__).resolve().parent.parent
if str(_bootstrap_root) not in sys.path:
    sys.path.append(str(_bootstrap_root))
from config import PathConfig
project_root = PathConfig.ROOT
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


CROP_CACHE_DIR = Path(__file__).resolve().parent / "cropped_tooth"


class ToothDatasetChaiCV(Dataset):
    """Like train_chai.py's ToothDatasetChai, but (a) reads from the one combined GroupKFold CSV,
    using each row's own orig_split (not a single Dataset-level split) to locate the right precrop
    subtree, and (b) reads precomputed tooth crops from ResNet/cropped_tooth/{orig_split}/{jaw}/
    {stem}_{fdi}.jpg (see fast_dataset.py's docstring for why this is safe for training - the
    sub-pixel crop-boundary difference from the int()-truncated box doesn't matter once GroupKFold
    means every number in the paper is being recomputed from scratch anyway) instead of decoding
    and cropping the full arch photo on every __getitem__ call, which made a single fold take
    multiple hours (confirmed: 2026-10-30, fold 0 ran ~3 hours without finishing even one epoch).
    Multi-worker DataLoader (num_workers>0) was tried as the fix first but hit a Windows-specific
    paging-file error when several worker processes each load the CUDA DLL stack at once - the
    precrop cache sidesteps that entirely by making each __getitem__ cheap enough that num_workers=0
    is fast on its own. (c) Caches each crop's raw JPEG BYTES (not decoded pixels) in a dict keyed
    by path, populated lazily on first access and reused for the rest of this fold's ~45-epoch run:
    a fold's full train+val crop set is ~755k files averaging ~10KB each (~7GB total) - fits RAM
    easily as compressed bytes, but decoded-pixel caching (~200KB/image) would need 100GB+. This
    was added after fold 0's real run still took ~50 wall-clock hours for 45 epochs despite the
    precrop cache (confirmed CPU-bound on cold per-epoch disk reads of ~75k distinct files, not
    GPU-bound) - caching bytes removes the repeat disk I/O from epoch 2 onward; decode-from-bytes
    still happens every call, but that alone was never the bottleneck."""

    def __init__(self, csv_path, dataset_dir, role_filter, transform=None, max_rows=None):
        self.rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                class12 = row.get('class12')
                if class12 is None or class12 == "" or not (0 <= int(class12) <= 11):
                    continue
                if not role_filter(row):
                    continue
                self.rows.append(row)
                if max_rows is not None and len(self.rows) >= max_rows:
                    break

        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.byte_cache = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_name = row['image_name']
        jaw = row['jaw']
        orig_split = row['orig_split']
        fdi_number = row['fdi_number']
        target_label = int(row['class12'])

        stem = Path(image_name).stem
        crop_path = CROP_CACHE_DIR / orig_split / jaw / f"{stem}_{fdi_number}.jpg"

        cached = self.byte_cache.get(crop_path)
        if cached is None:
            with open(crop_path, 'rb') as f:
                cached = f.read()
            self.byte_cache[crop_path] = cached
        image = Image.open(io.BytesIO(cached)).convert("RGB")

        if self.transform:
            img_tensor = self.transform(image)
        else:
            img_tensor = transforms.ToTensor()(image)

        meta = torch.tensor([
            float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2']),
        ], dtype=torch.float32)

        return img_tensor, meta, target_label


def train_one_fold(fold, dataset_dir, csv_path, model_save_path, runs_dir, epochs, batch_size, lr,
                    patience=10, resume=False, max_rows=None, num_workers=0):
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(runs_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(model_save_path).parent / "checkpoint.pth"

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def is_train_row(row):
        return row['cv_role'] == 'pool' and int(row['cv_fold']) != fold

    def is_val_row(row):
        return row['cv_role'] == 'pool' and int(row['cv_fold']) == fold

    print(f"\n=== Fold {fold} ===")
    print(f"Loading train rows from: {csv_path} (cv_role=pool, cv_fold != {fold})")
    train_dataset = ToothDatasetChaiCV(csv_path, dataset_dir, is_train_row, transform=train_transform, max_rows=max_rows)
    print(f"Loading val rows from: {csv_path} (cv_role=pool, cv_fold == {fold})")
    val_dataset = ToothDatasetChaiCV(csv_path, dataset_dir, is_val_row, transform=val_transform, max_rows=max_rows)

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    if train_size == 0 or val_size == 0:
        print("Error: empty train or val dataset.")
        return
    print(f"Train size: {train_size} | Val size: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                               pin_memory=True, persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True, persistent_workers=(num_workers > 0))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ToothPositionClassifierNoTheta(num_classes=12)
    model = model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []
    start_epoch = 0

    if resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt["best_acc"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        train_losses, val_losses = ckpt["train_losses"], ckpt["val_losses"]
        train_accs, val_accs = ckpt["train_accs"], ckpt["val_accs"]
        train_f1s, val_f1s = ckpt["train_f1s"], ckpt["val_f1s"]
        print(f"Resumed from {checkpoint_path}: starting at epoch {start_epoch + 1}, "
              f"best Val Acc so far {best_acc:.4f}, {epochs_no_improve}/{patience} epochs without improvement.")
    elif resume:
        print(f"--resume passed but no checkpoint found at {checkpoint_path}; starting fresh.")

    print(f"\nStarting ResNet18 Tooth Classifier (12 classes, GroupKFold fold {fold}) training...")
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_corrects = 0
        train_preds, train_labels_list = [], []

        for batch_idx, (inputs, metas, labels) in enumerate(train_loader):
            inputs, metas, labels = inputs.to(device), metas.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, metas)
            _, preds = torch.max(outputs, 1)
            loss = criterion(torch.log(outputs + 1e-15), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            train_preds.extend(preds.cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
        epoch_f1 = f1_score(train_labels_list, train_preds, average='macro')
        train_losses.append(epoch_loss); train_accs.append(epoch_acc.item()); train_f1s.append(epoch_f1)

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for batch_idx, (inputs, metas, labels) in enumerate(val_loader):
                inputs, metas, labels = inputs.to(device), metas.to(device), labels.to(device)
                outputs = model(inputs, metas)
                _, preds = torch.max(outputs, 1)
                loss = criterion(torch.log(outputs + 1e-15), labels)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        val_epoch_loss = val_loss / val_size
        val_epoch_acc = val_corrects.double() / val_size
        val_epoch_f1 = f1_score(val_labels_list, val_preds, average='macro')
        val_losses.append(val_epoch_loss); val_accs.append(val_epoch_acc.item()); val_f1s.append(val_epoch_f1)

        epoch_elapsed = time.time() - epoch_start
        print(f"[fold {fold}] Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f} F1: {val_epoch_f1:.4f} | "
              f"Time: {epoch_elapsed:.1f}s")

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Saved new best model to {model_save_path} (Val Acc: {best_acc:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  --> No improvement in Val Accuracy for {epochs_no_improve}/{patience} epochs.")

        torch.save({
            "epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "best_acc": best_acc, "epochs_no_improve": epochs_no_improve,
            "train_losses": train_losses, "val_losses": val_losses,
            "train_accs": train_accs, "val_accs": val_accs,
            "train_f1s": train_f1s, "val_f1s": val_f1s,
        }, checkpoint_path)

        if epochs_no_improve >= patience:
            print(f"\n[fold {fold}] Early stopping triggered at epoch {epoch+1} (no Val Accuracy improvement for {patience} epochs).")
            break

    print(f"\n[fold {fold}] Training completed. Best Validation Accuracy: {best_acc:.4f}")

    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'Loss History (CHAI GroupKFold fold {fold})'); plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accs, label='Train Acc', marker='o')
    plt.plot(epochs_range, val_accs, label='Val Acc', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title(f'Accuracy History (CHAI GroupKFold fold {fold})'); plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_f1s, label='Train F1', marker='o')
    plt.plot(epochs_range, val_f1s, label='Val F1', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('F1-Score (Macro)'); plt.title(f'F1-Score History (CHAI GroupKFold fold {fold})'); plt.legend()
    plt.tight_layout()
    plot_path = Path(runs_dir) / "metrics_history.png"
    plt.savefig(plot_path); plt.close()
    print(f"[fold {fold}] Saved metric plots to {plot_path.resolve()}")


def main():
    dataset_dir = project_root.parent / "dataset"
    csv_path = project_root / "ResNet" / "csv" / "features_groupkfold.csv"
    model_cv_dir = project_root / "ResNet" / "model_cv"
    runs_cv_dir = project_root / "ResNet" / "runs_cv"

    parser = argparse.ArgumentParser(description="Train CHAI under GroupKFold CV (one or all 10 folds).")
    parser.add_argument("--dataset_dir", type=str, default=str(dataset_dir))
    parser.add_argument("--csv_path", type=str, default=str(csv_path))
    parser.add_argument("--fold", type=int, default=None, help="Run just this fold (0-9); omit to loop all 10 sequentially.")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Smoke-test only: cap train/val rows at this many (per split) instead "
                             "of loading the full fold, so pipeline correctness can be checked in "
                             "minutes instead of hours. Omit for a real training run.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader worker processes (default: 0). Tried >0 to speed up "
                             "GroupKFold's 10x training runs, but on this machine it crashes with a "
                             "Windows paging-file error when several workers each load the CUDA DLL "
                             "stack at once - leave at 0 unless that's been fixed system-side. The "
                             "precrop cache (see ToothDatasetChaiCV) removes the actual need for it.")

    args = parser.parse_args()
    folds = [args.fold] if args.fold is not None else list(range(10))

    for fold in folds:
        train_one_fold(
            fold=fold,
            dataset_dir=args.dataset_dir,
            csv_path=args.csv_path,
            model_save_path=model_cv_dir / f"fold{fold}" / "best.pth",
            runs_dir=runs_cv_dir / f"fold{fold}",
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience,
            resume=args.resume, max_rows=args.max_rows, num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
