"""
Ablation counterpart to train.py, for the pooled (both-jaws, one model), no-rotation, no-theta
design (paper-review checklist item 0/1 follow-up - see functions/features/main_baseline.py's
docstring for the full reasoning): trains ONE combined 6-way (digit-only) classifier on both jaws
together, using functions/features/main_baseline.py's coordinates
(ResNet/tooth/csv_baseline/features_{train,val}.csv).

Reuses ToothPositionClassifier's ResNet backbone and classifier head from train.py as-is, but
swaps its meta_fc for a 4-input version (x1,y1,x2,y2, no theta) via ToothPositionClassifierNoTheta
below - train.py itself is untouched (still the base class every pooled/comp variant here
inherits from). train_nomirror.py / train_unified_mirror.py (the old B/C attempts that also
imported the unmodified 5-input ToothPositionClassifier) are superseded and moved to
backups/superseded_20260831/ along with train.py's own outputs (ResNet/tooth/model/,
ResNet/tooth/csv/) - variant A was dropped from the paper entirely, see project notes.

Usage:
    .venv/Scripts/python.exe ResNet/tooth/train_baseline.py [--resume]
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
from train import ToothPositionClassifier

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


class ToothPositionClassifierNoTheta(ToothPositionClassifier):
    """ToothPositionClassifier with a 4-dim meta input (x1,y1,x2,y2 - no theta). Theta was
    dropped from this ablation's features entirely (see main_baseline.py's docstring: it's
    PCA-angle-derived and was shown about as fragile to missing teeth as the coordinate rotation
    itself), so the meta_fc input layer shrinks from 5 to 4 - everything else (ResNet backbone,
    classifier head) is inherited unchanged."""

    def __init__(self, num_classes=6, pretrained=True):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        self.meta_fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU()
        )


class ToothDatasetCombined(Dataset):
    """Like train_unified_mirror.py's ToothDatasetCombined, but meta is [x1,y1,x2,y2] (4 dims,
    no theta) to match main_baseline.py's CSV columns."""

    def __init__(self, csv_path, dataset_dir, split, transform=None):
        self.rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fdi_digit = row.get('fdi_last_digit')
                if fdi_digit is not None and fdi_digit != "" and 1 <= int(fdi_digit) <= 6:
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
        fdi_digit = int(row['fdi_last_digit'])
        fdi_number = int(row['fdi_number'])
        target_label = fdi_digit - 1

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
            float(row['x1']),
            float(row['y1']),
            float(row['x2']),
            float(row['y2']),
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
    train_dataset = ToothDatasetCombined(csv_path=train_csv_path, dataset_dir=dataset_dir, split="train", transform=train_transform)

    print(f"Loading validation dataset from: {val_csv_path}")
    val_dataset = ToothDatasetCombined(csv_path=val_csv_path, dataset_dir=dataset_dir, split="val", transform=val_transform)

    train_size = len(train_dataset)
    val_size = len(val_dataset)

    if train_size == 0:
        print("Error: Train dataset is empty. Cannot train.")
        return
    if val_size == 0:
        print("Error: Validation dataset is empty. Cannot validate.")
        return

    print(f"Train size: {train_size} | Val size: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ToothPositionClassifierNoTheta(num_classes=6)
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

    print("\nStarting ResNet18 Tooth Classifier (6 classes, no-rotation, pooled) training...")
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        train_preds, train_labels_list = [], []

        for batch_idx, (inputs, metas, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            metas = metas.to(device)
            labels = labels.to(device)

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

        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())
        train_f1s.append(epoch_f1)

        model.eval()
        val_loss = 0.0
        val_corrects = 0

        val_preds, val_labels_list = [], []

        with torch.no_grad():
            for batch_idx, (inputs, metas, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                metas = metas.to(device)
                labels = labels.to(device)

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

        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc.item())
        val_f1s.append(val_epoch_f1)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f} | "
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
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_acc": best_acc,
            "epochs_no_improve": epochs_no_improve,
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
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss History (No-Rotation Pooled Model)')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accs, label='Train Acc', marker='o')
    plt.plot(epochs_range, val_accs, label='Val Acc', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy History (No-Rotation Pooled Model)')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_f1s, label='Train F1', marker='o')
    plt.plot(epochs_range, val_f1s, label='Val F1', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score (Macro)')
    plt.title('F1-Score History (No-Rotation Pooled Model)')
    plt.legend()

    plt.tight_layout()
    plot_path = Path(runs_dir) / "metrics_history.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved metric plots to {plot_path.resolve()}")

    print("\nGenerating final confusion matrix using best weights...")
    if Path(model_save_path).exists():
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()

        final_val_preds, final_val_labels = [], []
        with torch.no_grad():
            for inputs, metas, labels in val_loader:
                inputs = inputs.to(device)
                metas = metas.to(device)
                outputs = model(inputs, metas)
                _, preds = torch.max(outputs, 1)
                final_val_preds.extend(preds.cpu().numpy())
                final_val_labels.extend(labels.numpy())

        display_labels = ["1", "2", "3", "4", "5", "6"]
        cm = confusion_matrix(final_val_labels, final_val_preds, labels=list(range(6)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

        fig, ax = plt.subplots(figsize=(7, 7))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title("Confusion Matrix (Best No-Rotation Pooled Model)")
        plt.tight_layout()

        cm_path = Path(runs_dir) / "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion matrix plot saved at: {cm_path.resolve()}")


def main():
    dataset_dir = project_root.parent / "dataset"
    csv_dir = project_root / "ResNet" / "tooth" / "csv_baseline"
    model_dir = project_root / "ResNet" / "tooth" / "model_baseline"
    runs_dir = project_root / "ResNet" / "tooth" / "runs_baseline"

    parser = argparse.ArgumentParser(description="Train a combined (both jaws, no-rotation, no-theta, unified mirror) 6-way tooth digit classifier.")
    parser.add_argument("--dataset_dir", type=str, default=str(dataset_dir),
                        help="Path to the dataset folder")
    parser.add_argument("--csv_dir", type=str, default=str(csv_dir),
                        help="Directory holding features_{train,val}.csv (default: ResNet/tooth/csv_baseline)")
    parser.add_argument("--model_dir", type=str, default=str(model_dir),
                        help="Directory to save best.pth into (default: ResNet/tooth/model_baseline)")
    parser.add_argument("--runs_dir", type=str, default=str(runs_dir),
                        help="Directory to save metric plots/confusion matrix into (default: ResNet/tooth/runs_baseline)")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of epochs to train (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training (default: 128)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience in epochs (default: 10)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from {model_dir}/checkpoint.pth if it exists, instead of starting fresh")

    args = parser.parse_args()

    args_csv_dir = Path(args.csv_dir)
    args_model_dir = Path(args.model_dir)
    args_runs_dir = Path(args.runs_dir)

    train_csv_path = args_csv_dir / "features_train.csv"
    val_csv_path = args_csv_dir / "features_val.csv"
    model_save_path = args_model_dir / "best.pth"

    train_model(
        dataset_dir=args.dataset_dir,
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        model_save_path=model_save_path,
        runs_dir=args_runs_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
