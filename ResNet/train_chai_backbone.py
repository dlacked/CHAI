"""
Backbone-architecture ablation for CHAI: identical to train_chai.py (same ToothDatasetChai, same
4-dim meta-vector branch, same hyperparameters/augmentation/early-stopping) except the ResNet18
image branch is swapped for a different torchvision backbone, selected via --backbone. Isolates how
much of the proposed model's accuracy is attributable to the specific ResNet18 choice vs. backbone
capacity/architecture family in general.

Backbones supported (all torchvision, ImageNet-pretrained by default unless --no_pretrained is
also passed - kept orthogonal to the separate ImageNet-pretrained-weight ablation in
train_chai_no_pretrain.py):
    resnet18        (512-dim features)  - matches the production model exactly, useful as a sanity check
    resnet34        (512-dim features)  - same family, deeper
    resnet50        (2048-dim features) - same family, much higher capacity (bottleneck blocks)
    efficientnet_b0 (1280-dim features) - different family, designed for accuracy/efficiency trade-off
    mobilenet_v3_large (960-dim features) - different family, designed for low-latency deployment
    convnext_tiny   (768-dim features)  - different family, modern conv architecture

The classifier head's input width automatically follows each backbone's own feature dimension
(backbone_dim + 256 from the meta branch) - not hardcoded to 768 the way ToothPositionClassifier
implicitly is for ResNet18/34 specifically.

Deliberately a separate file (not a --backbone flag added to train_chai.py itself) per this
project's standing preference for ablation variants to live in their own sibling scripts. Writes to
model_{backbone}/ and runs_{backbone}/ - train_chai.py and model/ are left untouched.

Usage:
    .venv/Scripts/python.exe ResNet/train_chai_backbone.py --backbone resnet34
    .venv/Scripts/python.exe ResNet/train_chai_backbone.py --backbone resnet50
    .venv/Scripts/python.exe ResNet/train_chai_backbone.py --backbone efficientnet_b0
    .venv/Scripts/python.exe ResNet/train_chai_backbone.py --backbone mobilenet_v3_large
    .venv/Scripts/python.exe ResNet/train_chai_backbone.py --backbone convnext_tiny
"""
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tooth_dir = Path(__file__).resolve().parent
if str(tooth_dir) not in sys.path:
    sys.path.append(str(tooth_dir))
from fast_dataset import ToothDatasetChaiFast as ToothDatasetChai  # noqa: E402

_bootstrap_root = Path(__file__).resolve().parent.parent
if str(_bootstrap_root) not in sys.path:
    sys.path.append(str(_bootstrap_root))
from config import PathConfig
project_root = PathConfig.ROOT
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


def build_backbone(name, pretrained=True):
    """Returns (backbone_module_with_classifier_head_removed, feature_dim)."""
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        feat_dim = m.classifier[-1].in_features
        m.classifier = nn.Identity()
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        feat_dim = m.classifier[0].in_features
        m.classifier = nn.Identity()
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        feat_dim = m.classifier[-1].in_features
        m.classifier = nn.Sequential(m.classifier[0], nn.Flatten(1))
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return m, feat_dim


class ToothPositionClassifierBackbone(nn.Module):
    """Like ToothPositionClassifierNoTheta (4-dim meta, no theta, 12-way output for CHAI), but
    with a pluggable image backbone instead of a hardcoded ResNet18."""

    def __init__(self, backbone_name, num_classes=12, pretrained=True):
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone_name, pretrained=pretrained)
        self.meta_fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, meta):
        img_features = self.backbone(img)
        meta_features = self.meta_fc(meta)
        combined = torch.cat((img_features, meta_features), dim=1)
        logits = self.classifier(combined)
        return torch.softmax(logits, dim=1)


def train_model(backbone_name, dataset_dir, train_csv_path, val_csv_path, model_save_path, runs_dir,
                 epochs, batch_size, lr, patience=10, resume=False, pretrained=True):
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
    train_dataset = ToothDatasetChai(csv_path=train_csv_path, dataset_dir=dataset_dir, split="train", transform=train_transform)
    print(f"Loading validation dataset from: {val_csv_path}")
    val_dataset = ToothDatasetChai(csv_path=val_csv_path, dataset_dir=dataset_dir, split="val", transform=val_transform)

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

    model = ToothPositionClassifierBackbone(backbone_name, num_classes=12, pretrained=pretrained)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Backbone: {backbone_name} | Total params: {n_params:,}")

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

    print(f"\nStarting ResNet-family-agnostic Tooth Classifier (backbone={backbone_name}) training...")
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
            if (batch_idx + 1) % 50 == 0:
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
            for inputs, metas, labels in val_loader:
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
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'Loss History (CHAI, {backbone_name})'); plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accs, label='Train Acc', marker='o')
    plt.plot(epochs_range, val_accs, label='Val Acc', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title(f'Accuracy History (CHAI, {backbone_name})'); plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_f1s, label='Train F1', marker='o')
    plt.plot(epochs_range, val_f1s, label='Val F1', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('F1-Score (Macro)'); plt.title(f'F1-Score History (CHAI, {backbone_name})'); plt.legend()
    plt.tight_layout()
    plot_path = Path(runs_dir) / "metrics_history.png"
    plt.savefig(plot_path); plt.close()
    print(f"Saved metric plots to {plot_path.resolve()}")


def main():
    dataset_dir = project_root.parent / "dataset"
    csv_dir = project_root / "ResNet" / "csv"

    parser = argparse.ArgumentParser(description="Backbone-architecture ablation for CHAI.")
    parser.add_argument("--backbone", type=str, required=True,
                         choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0",
                                  "mobilenet_v3_large", "convnext_tiny"])
    parser.add_argument("--dataset_dir", type=str, default=str(dataset_dir))
    parser.add_argument("--csv_dir", type=str, default=str(csv_dir))
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--runs_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")

    args = parser.parse_args()
    model_dir = Path(args.model_dir) if args.model_dir else project_root / "ResNet" / f"model_{args.backbone}"
    runs_dir = Path(args.runs_dir) if args.runs_dir else project_root / "ResNet" / f"runs_{args.backbone}"
    args_csv_dir = Path(args.csv_dir)

    train_model(
        backbone_name=args.backbone,
        dataset_dir=args.dataset_dir,
        train_csv_path=args_csv_dir / "features_train.csv",
        val_csv_path=args_csv_dir / "features_val.csv",
        model_save_path=model_dir / "best.pth",
        runs_dir=runs_dir,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience,
        resume=args.resume, pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    main()
