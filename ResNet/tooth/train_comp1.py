"""
This is Comp1 (paper label) - ablation counterpart to train_baseline.py (Baseline, "CHAI (Ours)"):
"what role do coordinates actually play." Trains the SAME pooled (both-jaws-combined, 6-way)
setup as Baseline, but with NO coordinate/meta features at all - image only. Reuses
train_baseline.py's csv_baseline data and ToothDatasetCombined as-is (x1/y1/x2/y2 columns are read
but simply never handed to the model), so Comp1 differs from Baseline in exactly one way:
ToothPositionClassifierImageOnly drops the meta_fc branch and its concatenation entirely,
classifying straight off the ResNet image features.

Comparing Comp1 against Baseline isolates "how much do coordinates help at all" - compare against
Comp2 (train_baseline.py run against csv_comp2 instead, see
functions/features/main_comp2.py) for "does mirroring specifically matter," and Comp3
(train_comp3.py) for "does the separate geometric-tens-assignment + Hungarian pipeline matter."
The old variant-E attempt at "does mirroring matter" (train_e12way.py, per-jaw + baseline A's PCA
rotation) is superseded by Comp2 and moved to backups/superseded_20260831/.

train_baseline.py and its outputs (ResNet/tooth/model_baseline/) are left completely alone.

Usage:
    .venv/Scripts/python.exe ResNet/tooth/train_imageonly.py [--resume]
"""
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tooth_dir = Path(__file__).resolve().parent
if str(tooth_dir) not in sys.path:
    sys.path.append(str(tooth_dir))
from train import ToothPositionClassifier
from train_baseline import ToothDatasetCombined

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


class ToothPositionClassifierImageOnly(ToothPositionClassifier):
    """ToothPositionClassifier with the meta_fc branch and its concatenation removed entirely -
    classifies straight off the ResNet's own image features (no coordinates/theta at all). Only
    self.classifier's input width changes (num_ftrs instead of num_ftrs+256); the ResNet backbone
    is inherited unchanged."""

    def __init__(self, num_classes=6, pretrained=True):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        num_ftrs = self.resnet.fc.in_features if hasattr(self.resnet.fc, 'in_features') else 512
        # resnet.fc was already replaced with nn.Identity() in the base __init__, so recover the
        # feature width from resnet18's known layer4 output channels instead.
        num_ftrs = 512
        del self.meta_fc
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, meta=None):
        img_features = self.resnet(img)
        logits = self.classifier(img_features)
        return torch.softmax(logits, dim=1)


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
    if train_size == 0 or val_size == 0:
        print("Error: empty train or val dataset.")
        return
    print(f"Train size: {train_size} | Val size: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ToothPositionClassifierImageOnly(num_classes=6)
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

    print("\nStarting ResNet18 Tooth Classifier (6 classes, image-only, pooled) training...")
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        train_preds, train_labels_list = [], []

        for batch_idx, (inputs, metas, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
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
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
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
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss History (Image-Only Pooled Model)'); plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accs, label='Train Acc', marker='o')
    plt.plot(epochs_range, val_accs, label='Val Acc', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy History (Image-Only Pooled Model)'); plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_f1s, label='Train F1', marker='o')
    plt.plot(epochs_range, val_f1s, label='Val F1', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('F1-Score (Macro)'); plt.title('F1-Score History (Image-Only Pooled Model)'); plt.legend()
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
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                final_val_preds.extend(preds.cpu().numpy())
                final_val_labels.extend(labels.numpy())
        cm = confusion_matrix(final_val_labels, final_val_preds, labels=list(range(6)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["1", "2", "3", "4", "5", "6"])
        fig, ax = plt.subplots(figsize=(7, 7))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title("Confusion Matrix (Best Image-Only Pooled Model)")
        plt.tight_layout()
        cm_path = Path(runs_dir) / "confusion_matrix.png"
        plt.savefig(cm_path); plt.close()
        print(f"Confusion matrix plot saved at: {cm_path.resolve()}")


def main():
    dataset_dir = project_root.parent / "dataset"
    csv_dir = project_root / "ResNet" / "tooth" / "csv_baseline"
    model_dir = project_root / "ResNet" / "tooth" / "model_comp1"
    runs_dir = project_root / "ResNet" / "tooth" / "runs_comp1"

    parser = argparse.ArgumentParser(description="Train a combined (both jaws), image-only (no coordinates) 6-way tooth digit classifier.")
    parser.add_argument("--dataset_dir", type=str, default=str(dataset_dir))
    parser.add_argument("--csv_dir", type=str, default=str(csv_dir),
                        help="Reuses csv_baseline (x1..y2 columns present but unused)")
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
