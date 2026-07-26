import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COMPLEXITY_DIR = Path(__file__).resolve().parent
if str(COMPLEXITY_DIR) not in sys.path:
    sys.path.append(str(COMPLEXITY_DIR))

from dataset import ArchComplexityDataset, pad_collate
from model import ArchComplexityTransformer

DISPLAY_LABELS = ["1", "2", "3"]


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


def evaluate_test(model, cache_dir, jaw, device, criterion, batch_size, runs_dir):
    test_cache = Path(cache_dir) / f"{jaw}_test.pt"
    if not test_cache.exists():
        print(f"No test cache found ({test_cache}); skipping test evaluation.")
        return

    test_dataset = ArchComplexityDataset(test_cache)
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
    plt.title(f"Test Confusion Matrix ({jaw.upper()} Arch Complexity)")
    plt.tight_layout()
    cm_path = Path(runs_dir) / f"{jaw}_test_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Test confusion matrix saved at: {cm_path.resolve()}")


def train_model(jaw, cache_dir, model_save_path, runs_dir, epochs, batch_size, lr, patience):
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(runs_dir).mkdir(parents=True, exist_ok=True)

    train_cache = Path(cache_dir) / f"{jaw}_train.pt"
    val_cache = Path(cache_dir) / f"{jaw}_val.pt"
    if not train_cache.exists() or not val_cache.exists():
        print(f"Error: cache files not found ({train_cache}, {val_cache}). Run build_dataset.py first.")
        return

    train_dataset = ArchComplexityDataset(train_cache)
    val_dataset = ArchComplexityDataset(val_cache)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: empty train/val dataset.")
        return

    print(f"Train arches: {len(train_dataset)} | Val arches: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ArchComplexityTransformer(num_classes=len(DISPLAY_LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    print(f"\nStarting Arch Complexity Transformer training for {jaw.upper()} jaw...")
    for epoch in range(epochs):
        train_loss, train_acc, train_f1, _, _ = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc, val_f1, _, _ = run_epoch(model, val_loader, device, criterion, optimizer=None)

        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)
        train_f1s.append(train_f1); val_f1s.append(val_f1)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Saved new best model to {model_save_path} (Val Acc: {best_acc:.4f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  --> No improvement in Val Loss for {epochs_no_improve}/{patience} epochs.")
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1} (no Val Loss improvement for {patience} epochs).")
                break

    print(f"\nTraining completed. Best Validation Accuracy: {best_acc:.4f}")

    # Plot and save metrics history
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f'Loss History ({jaw.upper()} Arch Complexity)'); plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accs, label='Train Acc', marker='o')
    plt.plot(epochs_range, val_accs, label='Val Acc', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title(f'Accuracy History ({jaw.upper()} Arch Complexity)'); plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_f1s, label='Train F1', marker='o')
    plt.plot(epochs_range, val_f1s, label='Val F1', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('F1-Score (Macro)')
    plt.title(f'F1-Score History ({jaw.upper()} Arch Complexity)'); plt.legend()

    plt.tight_layout()
    plot_path = Path(runs_dir) / f"{jaw}_metrics_history.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved metric plots to {plot_path.resolve()}")

    # Final confusion matrix using best weights
    print("\nGenerating final confusion matrix using best weights...")
    if Path(model_save_path).exists():
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        _, _, _, final_val_preds, final_val_labels = run_epoch(model, val_loader, device, criterion, optimizer=None)

        cm = confusion_matrix(final_val_labels, final_val_preds, labels=list(range(len(DISPLAY_LABELS))))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DISPLAY_LABELS)

        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title(f"Confusion Matrix (Best {jaw.upper()} Arch Complexity)")
        plt.tight_layout()

        cm_path = Path(runs_dir) / f"{jaw}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion matrix plot saved at: {cm_path.resolve()}")

        print("\nEvaluating on held-out test split...")
        evaluate_test(model, cache_dir, jaw, device, criterion, batch_size, runs_dir)


def main():
    cache_dir_default = COMPLEXITY_DIR / "cache"
    model_dir_default = COMPLEXITY_DIR / "model"
    runs_dir_default = COMPLEXITY_DIR / "runs"

    parser = argparse.ArgumentParser(description="Train the arch-level Transformer complexity classifier.")
    parser.add_argument("--jaw", type=str, default="lower", choices=("lower", "upper"),
                         help="Jaw model to train: lower or upper (default: lower)")
    parser.add_argument("--cache_dir", type=str, default=str(cache_dir_default),
                         help="Directory with {jaw}_{train,val,test}.pt caches from build_dataset.py")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size in arches (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience in epochs (default: 10)")

    args = parser.parse_args()

    model_save_path = Path(model_dir_default) / f"{args.jaw}_best.pth"

    train_model(
        jaw=args.jaw,
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
