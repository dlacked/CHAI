"""
Main Arch Complexity Transformer training script - pooled (both-jaws-combined): trains ONE model
on lower+upper arches together. Promoted to this name 2026-08-31 after a held-out test confirmed
pooling doesn't hurt accuracy (99.83% test, vs. 99.3-100% for the old separate per-jaw models on
a different, PCA-rotated coordinate scheme - not a strictly controlled comparison, but close
enough given both are already near-ceiling). The old per-jaw training script (`train.py --jaw
lower/upper`, PCA-rotated coordinates) and its outputs are retired to
backups/superseded_20260831/Transformer_complexity_{train_perjaw.py,model_perjaw,runs_perjaw}/.

Feasible at all because build_dataset.py's coordinate source (csv_comp3 - see that file's
csv_dir_default comment) keeps Baseline's Y-flip, which aligns upper/lower jaw coordinate ranges
into one shared convention - the same reason Baseline's tooth-digit ResNet classifier can be
pooled (see functions/features/coords_baseline.py's docstring). Before that coordinate swap,
lower/upper Complexity models had no shared coordinate convention and pooling wasn't meaningful.

The model itself (ArchComplexityTransformer) needs no changes - it never took a jaw indicator as
input, it's already jaw-agnostic; only the DATA needs combining.

Usage:
    .venv/Scripts/python.exe Transformer/complexity/train.py
"""
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COMPLEXITY_DIR = Path(__file__).resolve().parent
if str(COMPLEXITY_DIR) not in sys.path:
    sys.path.append(str(COMPLEXITY_DIR))

from dataset import pad_collate
from model import ArchComplexityTransformer

GEOM_DIM = 4
DISPLAY_LABELS = ["1", "2", "3"]


class PooledArchComplexityDataset(Dataset):
    """Like dataset.py's ArchComplexityDataset, but loads and concatenates MULTIPLE per-jaw
    cache files (e.g. lower_train.pt + upper_train.pt) into one combined dataset - the model
    itself doesn't need to know which jaw an arch came from, so this is just concatenation, no
    extra field."""

    def __init__(self, cache_paths):
        self.sequences = []
        for cache_path in cache_paths:
            cache_path = Path(cache_path)
            if not cache_path.exists():
                print(f"Warning: cache not found, skipping: {cache_path}")
                continue
            self.sequences.extend(torch.load(cache_path))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        return item["geom"], item["complexity"]


def run_epoch(model, loader, device, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    total_arches = 0
    all_preds, all_labels = [], []

    with torch.enable_grad() if train_mode else torch.no_grad():
        for geom, target, key_padding_mask in loader:
            geom = geom.to(device)
            target = target.to(device)
            key_padding_mask = key_padding_mask.to(device)

            if train_mode:
                optimizer.zero_grad()

            logits = model(geom, key_padding_mask)
            loss = criterion(logits, target)

            if train_mode:
                loss.backward()
                optimizer.step()

            n = target.size(0)
            total_loss += loss.item() * n
            total_arches += n

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    avg_loss = total_loss / max(total_arches, 1)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(total_arches, 1)
    f1 = f1_score(all_labels, all_preds, average='macro') if total_arches > 0 else 0.0
    return avg_loss, acc, f1, all_preds, all_labels


def evaluate_test(model, cache_dir, device, criterion, batch_size, runs_dir):
    test_dataset = PooledArchComplexityDataset([
        Path(cache_dir) / "lower_test.pt", Path(cache_dir) / "upper_test.pt",
    ])
    if len(test_dataset) == 0:
        print("Empty test dataset; skipping test evaluation.")
        return

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    test_loss, test_acc, test_f1, test_preds, test_labels = run_epoch(model, test_loader, device, criterion, optimizer=None)
    print(f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} F1: {test_f1:.4f}")

    cm = confusion_matrix(test_labels, test_preds, labels=list(range(len(DISPLAY_LABELS))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DISPLAY_LABELS)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Test Confusion Matrix (Pooled Arch Complexity)")
    plt.tight_layout()
    cm_path = Path(runs_dir) / "test_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Test confusion matrix saved at: {cm_path.resolve()}")


def train_model(cache_dir, model_save_path, runs_dir, epochs, batch_size, lr, patience):
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(runs_dir).mkdir(parents=True, exist_ok=True)

    train_dataset = PooledArchComplexityDataset([
        Path(cache_dir) / "lower_train.pt", Path(cache_dir) / "upper_train.pt",
    ])
    val_dataset = PooledArchComplexityDataset([
        Path(cache_dir) / "lower_val.pt", Path(cache_dir) / "upper_val.pt",
    ])

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: empty train/val dataset.")
        return

    print(f"Train arches: {len(train_dataset)} | Val arches: {len(val_dataset)} (both jaws pooled)")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ArchComplexityTransformer(geom_dim=GEOM_DIM, num_classes=len(DISPLAY_LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    print("\nStarting Arch Complexity Transformer (pooled, both jaws) training...")
    for epoch in range(epochs):
        train_loss, train_acc, train_f1, _, _ = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc, val_f1, _, _ = run_epoch(model, val_loader, device, criterion, optimizer=None)

        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)
        train_f1s.append(train_f1); val_f1s.append(val_f1)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Saved new best model to {model_save_path} (Val Acc: {best_acc:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  --> No improvement in Val Accuracy for {epochs_no_improve}/{patience} epochs.")
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1} (no Val Accuracy improvement for {patience} epochs).")
                break

    print(f"\nTraining completed. Best Validation Accuracy: {best_acc:.4f}")

    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Loss History (Pooled Arch Complexity)'); plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accs, label='Train Acc', marker='o')
    plt.plot(epochs_range, val_accs, label='Val Acc', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Accuracy History (Pooled Arch Complexity)'); plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_f1s, label='Train F1', marker='o')
    plt.plot(epochs_range, val_f1s, label='Val F1', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('F1-Score (Macro)')
    plt.title('F1-Score History (Pooled Arch Complexity)'); plt.legend()

    plt.tight_layout()
    plot_path = Path(runs_dir) / "metrics_history.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved metric plots to {plot_path.resolve()}")

    print("\nGenerating final confusion matrix using best weights...")
    if Path(model_save_path).exists():
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        _, _, _, final_val_preds, final_val_labels = run_epoch(model, val_loader, device, criterion, optimizer=None)

        cm = confusion_matrix(final_val_labels, final_val_preds, labels=list(range(len(DISPLAY_LABELS))))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DISPLAY_LABELS)

        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title("Confusion Matrix (Best Pooled Arch Complexity)")
        plt.tight_layout()

        cm_path = Path(runs_dir) / "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion matrix plot saved at: {cm_path.resolve()}")

        print("\nEvaluating on held-out test split...")
        evaluate_test(model, cache_dir, device, criterion, batch_size, runs_dir)


def main():
    cache_dir_default = COMPLEXITY_DIR / "cache"
    model_dir_default = COMPLEXITY_DIR / "model"
    runs_dir_default = COMPLEXITY_DIR / "runs"

    parser = argparse.ArgumentParser(description="Train the pooled (both-jaws) Arch Complexity Transformer.")
    parser.add_argument("--cache_dir", type=str, default=str(cache_dir_default),
                         help="Directory with lower_/upper_{train,val,test}.pt caches from build_dataset.py")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size in arches (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience in epochs (default: 20)")

    args = parser.parse_args()

    model_save_path = Path(model_dir_default) / "best.pth"

    train_model(
        cache_dir=args.cache_dir,
        model_save_path=model_save_path,
        runs_dir=runs_dir_default,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
