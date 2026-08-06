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

ARCH_DIR = Path(__file__).resolve().parent
if str(ARCH_DIR) not in sys.path:
    sys.path.append(str(ARCH_DIR))

from dataset import ArchSequenceDataset, pad_collate
from model import ArchToothTransformer


def flatten_valid(logits, targets):
    """Flattens (B,K,C)/(B,K) and drops padded (-100) positions."""
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = targets.reshape(-1)
    valid = targets_flat != -100
    return logits_flat[valid], targets_flat[valid]


GRAD_CLIP_MAX_NORM = 1.0


def run_epoch(model, loader, device, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    total_teeth = 0
    all_preds, all_labels = [], []

    with torch.enable_grad() if train_mode else torch.no_grad():
        for geom, img_vec, prob_vec, target, key_padding_mask in loader:
            geom = geom.to(device)
            img_vec = img_vec.to(device)
            prob_vec = prob_vec.to(device)
            target = target.to(device)
            key_padding_mask = key_padding_mask.to(device)

            if train_mode:
                optimizer.zero_grad()

            logits = model(geom, img_vec, prob_vec, key_padding_mask)
            logits_flat, target_flat = flatten_valid(logits, target)
            loss = criterion(logits_flat, target_flat)

            if train_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                optimizer.step()

            n = target_flat.size(0)
            total_loss += loss.item() * n
            total_teeth += n

            preds = torch.argmax(logits_flat, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target_flat.cpu().numpy())

    avg_loss = total_loss / max(total_teeth, 1)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(total_teeth, 1)
    f1 = f1_score(all_labels, all_preds, average='macro') if total_teeth > 0 else 0.0
    return avg_loss, acc, f1, all_preds, all_labels


def train_model(jaw, cache_dir, model_save_path, runs_dir, epochs, batch_size, lr, patience):
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(runs_dir).mkdir(parents=True, exist_ok=True)

    train_cache = Path(cache_dir) / f"{jaw}_train.pt"
    val_cache = Path(cache_dir) / f"{jaw}_val.pt"
    if not train_cache.exists() or not val_cache.exists():
        print(f"Error: cache files not found ({train_cache}, {val_cache}). Run build_dataset.py first.")
        return

    train_dataset = ArchSequenceDataset(train_cache)
    val_dataset = ArchSequenceDataset(val_cache)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: empty train/val dataset.")
        return

    print(f"Train arches: {len(train_dataset)} | Val arches: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ArchToothTransformer(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Val loss oscillates once the model is near convergence at a fixed lr - halving it after a
    # few stagnant epochs (patience shorter than early-stopping's) damps that before training
    # gives up entirely.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_acc = 0.0
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    print(f"\nStarting Arch Transformer training for {jaw.upper()} jaw...")
    for epoch in range(epochs):
        train_loss, train_acc, train_f1, _, _ = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc, val_f1, _, _ = run_epoch(model, val_loader, device, criterion, optimizer=None)

        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)
        train_f1s.append(train_f1); val_f1s.append(val_f1)

        lr_before = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        lr_after = optimizer.param_groups[0]['lr']
        if lr_after < lr_before:
            print(f"  --> Val loss plateaued; reducing LR {lr_before:.2e} -> {lr_after:.2e}")

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
              f"LR: {lr_after:.2e}")

        # Save-best and early-stopping both track validation accuracy - the metric the saved
        # checkpoint is actually judged on - so the two log lines below can never contradict
        # each other the way they could when one tracked accuracy and the other tracked loss.
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

    # Plot and save metrics history
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f'Loss History ({jaw.upper()} Arch Transformer)'); plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accs, label='Train Acc', marker='o')
    plt.plot(epochs_range, val_accs, label='Val Acc', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title(f'Accuracy History ({jaw.upper()} Arch Transformer)'); plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_f1s, label='Train F1', marker='o')
    plt.plot(epochs_range, val_f1s, label='Val F1', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('F1-Score (Macro)')
    plt.title(f'F1-Score History ({jaw.upper()} Arch Transformer)'); plt.legend()

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

        display_labels = ["1", "2", "3", "4", "5", "6"]
        cm = confusion_matrix(final_val_labels, final_val_preds, labels=list(range(6)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

        fig, ax = plt.subplots(figsize=(7, 7))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title(f"Confusion Matrix (Best {jaw.upper()} Arch Transformer)")
        plt.tight_layout()

        cm_path = Path(runs_dir) / f"{jaw}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion matrix plot saved at: {cm_path.resolve()}")


def main():
    cache_dir_default = ARCH_DIR / "cache"
    model_dir_default = ARCH_DIR / "model"
    runs_dir_default = ARCH_DIR / "runs"

    parser = argparse.ArgumentParser(description="Train the arch-level Transformer tooth-number refiner.")
    parser.add_argument("--jaw", type=str, default="lower", choices=("lower", "upper"),
                         help="Jaw model to train: lower or upper (default: lower)")
    parser.add_argument("--cache_dir", type=str, default=str(cache_dir_default),
                         help="Directory with {jaw}_{train,val}.pt caches from build_dataset.py")
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
