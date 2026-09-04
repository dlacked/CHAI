"""
Comp3 (paper label): pooled (both-jaws-combined) 12-way classifier (this jaw's 2 quadrants x 6
digits - class12, same scheme as main_e12way.py (moved to backups/superseded_20260831/)'s old
E/Comp2 ablation) on Baseline's coordinate convention minus the X-mirror
(functions/features/main_comp3.py's ResNet/tooth/csv_comp3/features_{train,val,test}.csv).
Tests "how much does skipping the separate geometric-tens-assignment + Hungarian post-process
pipeline cost" - a Comp3 prediction (class12) plus the already-known jaw (upper/lower)
reconstructs the full FDI number directly, no Hungarian duplicate-resolution step involved.
Evaluation (not this file - see project notes) should score the model's raw per-tooth argmax as
the final answer, unlike Baseline/Comp1/Comp2 which all still go through Hungarian.

RENAMED 2026-08-31: this used to read ResNet/tooth/csv_comp2/ (shared with Comp2, back when Comp2
was pooled with the same Y-flip-kept/X-mirror-dropped coordinates as Comp3). Comp2 was then
redefined to fully-raw coordinates + separate per-jaw models (see ResNet/tooth/train_comp2.py),
so csv_comp2/ now means something different - this reads csv_comp3/ instead, which is
Comp3-dedicated and unchanged in every other respect.

Reuses ToothPositionClassifier's ResNet backbone/classifier-head shape from train.py via
train_baseline.py's ToothPositionClassifierNoTheta (4-dim meta, no theta - just constructed with
num_classes=12 here instead of 6). Reuses train_baseline.py's ToothDatasetCombined shape almost
exactly - only the label column differs (class12 instead of fdi_last_digit), same as
train_e12way.py's relationship to train.py's ToothDataset.

train_baseline.py and its outputs (ResNet/tooth/model_baseline/) are left completely alone.

Usage:
    .venv/Scripts/python.exe ResNet/tooth/train_comp3.py [--resume]
"""
import sys
import os
import json
import argparse
import csv
from pathlib import Path
import numpy as np
import cv2
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
from train_baseline import ToothPositionClassifierNoTheta

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


class ToothDatasetComp3(Dataset):
    """Like train_baseline.py's ToothDatasetCombined, but the label is class12 (0-11: this jaw's
    2 quadrants x 6 digits) instead of fdi_last_digit-1 (0-5) - everything else (per-row jaw
    resolution, crop logic, 4-dim meta) is identical."""

    def __init__(self, csv_path, dataset_dir, split, transform=None):
        self.rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                class12 = row.get('class12')
                if class12 is not None and class12 != "" and 0 <= int(class12) <= 11:
                    self.rows.append(row)

        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.transform = transform
        self.crop_cache = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_name = row['image_name']
        jaw = row['jaw']
        fdi_number = int(row['fdi_number'])
        target_label = int(row['class12'])

        cache_key = (jaw, image_name, fdi_number)

        img_path = self.dataset_dir / self.split / "images" / jaw / image_name
        image = cv2.imread(str(img_path))
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        h, w = image.shape[:2]

        if cache_key in self.crop_cache:
            x1, y1, x2, y2 = self.crop_cache[cache_key]
        else:
            x1, y1, x2, y2 = 0, 0, w, h

            image_name_no_ext = os.path.splitext(image_name)[0]
            json_path = self.dataset_dir / self.split / "labels_json" / jaw / f"{image_name_no_ext}.json"

            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    for t in data.get("tooth", []):
                        if t.get("teeth_num") == fdi_number:
                            seg = t.get("segmentation", [])
                            if seg:
                                if isinstance(seg[0], list) and len(seg[0]) == 2:
                                    poly = np.array(seg)
                                else:
                                    poly = np.array(seg).reshape(-1, 2)

                                x_min, y_min = poly.min(axis=0)
                                x_max, y_max = poly.max(axis=0)

                                pad = 10
                                x1 = int(max(0, x_min - pad))
                                y1 = int(max(0, y_min - pad))
                                x2 = int(min(w, x_max + pad))
                                y2 = int(min(h, y_max + pad))
                                break
                except Exception as e:
                    print(f"Error loading GT json for crop: {e}")

            self.crop_cache[cache_key] = (x1, y1, x2, y2)

        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            cropped = np.zeros((224, 224, 3), dtype=np.uint8)

        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cropped_rgb)

        if self.transform:
            img_tensor = self.transform(pil_img)
        else:
            img_tensor = transforms.ToTensor()(pil_img)

        meta = torch.tensor([
            float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2']),
        ], dtype=torch.float32)

        return img_tensor, meta, target_label


def train_model(dataset_dir, train_csv_path, val_csv_path, model_save_path, runs_dir, epochs, batch_size, lr, patience=10, resume=False):
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

    print(f"Loading train dataset from: {train_csv_path}")
    train_dataset = ToothDatasetComp3(csv_path=train_csv_path, dataset_dir=dataset_dir, split="train", transform=train_transform)
    print(f"Loading validation dataset from: {val_csv_path}")
    val_dataset = ToothDatasetComp3(csv_path=val_csv_path, dataset_dir=dataset_dir, split="val", transform=val_transform)

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    if train_size == 0 or val_size == 0:
        print("Error: empty train or val dataset.")
        return
    print(f"Train size: {train_size} | Val size: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

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

    print("\nStarting ResNet18 Tooth Classifier (12 classes, no-mirror, pooled, Comp3) training...")
    for epoch in range(start_epoch, epochs):
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
            print(f"  [Train] Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f} | Running Acc: {running_corrects.double() / len(train_labels_list):.4f}")

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
                print(f"  [Val] Batch [{batch_idx+1}/{len(val_loader)}] | Loss: {loss.item():.4f} | Running Acc: {val_corrects.double() / len(val_labels_list):.4f}")

        val_epoch_loss = val_loss / val_size
        val_epoch_acc = val_corrects.double() / val_size
        val_epoch_f1 = f1_score(val_labels_list, val_preds, average='macro')
        val_losses.append(val_epoch_loss); val_accs.append(val_epoch_acc.item()); val_f1s.append(val_epoch_f1)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f} F1: {val_epoch_f1:.4f}")

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
            print(f"\nEarly stopping triggered at epoch {epoch+1} (no Val Accuracy improvement for {patience} epochs).")
            break

    print(f"\nTraining completed. Best Validation Accuracy: {best_acc:.4f}")

    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss History (Comp3: Pooled No-Mirror 12-way)'); plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accs, label='Train Acc', marker='o')
    plt.plot(epochs_range, val_accs, label='Val Acc', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy History (Comp3: Pooled No-Mirror 12-way)'); plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_f1s, label='Train F1', marker='o')
    plt.plot(epochs_range, val_f1s, label='Val F1', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('F1-Score (Macro)'); plt.title('F1-Score History (Comp3: Pooled No-Mirror 12-way)'); plt.legend()
    plt.tight_layout()
    plot_path = Path(runs_dir) / "metrics_history.png"
    plt.savefig(plot_path); plt.close()
    print(f"Saved metric plots to {plot_path.resolve()}")

    if Path(model_save_path).exists():
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()
        final_val_preds, final_val_labels = [], []
        with torch.no_grad():
            for inputs, metas, labels in val_loader:
                inputs, metas = inputs.to(device), metas.to(device)
                outputs = model(inputs, metas)
                _, preds = torch.max(outputs, 1)
                final_val_preds.extend(preds.cpu().numpy())
                final_val_labels.extend(labels.numpy())
        cm = confusion_matrix(final_val_labels, final_val_preds, labels=list(range(12)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(12)])
        fig, ax = plt.subplots(figsize=(9, 9))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title("Confusion Matrix (Best Comp3 Model)")
        plt.tight_layout()
        cm_path = Path(runs_dir) / "confusion_matrix.png"
        plt.savefig(cm_path); plt.close()
        print(f"Confusion matrix plot saved at: {cm_path.resolve()}")


def main():
    dataset_dir = project_root.parent / "dataset"
    csv_dir = project_root / "ResNet" / "tooth" / "csv_comp3"
    model_dir = project_root / "ResNet" / "tooth" / "model_comp3"
    runs_dir = project_root / "ResNet" / "tooth" / "runs_comp3"

    parser = argparse.ArgumentParser(description="Train Comp3: pooled, no-mirror, 12-way (up-to-tens-digit) tooth classifier - no Hungarian at eval time.")
    parser.add_argument("--dataset_dir", type=str, default=str(dataset_dir))
    parser.add_argument("--csv_dir", type=str, default=str(csv_dir))
    parser.add_argument("--model_dir", type=str, default=str(model_dir))
    parser.add_argument("--runs_dir", type=str, default=str(runs_dir))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    args_csv_dir = Path(args.csv_dir)
    args_model_dir = Path(args.model_dir)
    args_runs_dir = Path(args.runs_dir)

    train_model(
        dataset_dir=args.dataset_dir,
        train_csv_path=args_csv_dir / "features_train.csv",
        val_csv_path=args_csv_dir / "features_val.csv",
        model_save_path=args_model_dir / "best.pth",
        runs_dir=args_runs_dir,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
