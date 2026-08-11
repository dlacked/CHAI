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
from torchvision import models, transforms
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

class ToothDataset(Dataset):
    def __init__(self, csv_path, dataset_dir, jaw, split, transform=None):
        self.rows = []
        # Load CSV using built-in csv module
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fdi_digit = row.get('fdi_last_digit')
                if fdi_digit is not None and fdi_digit != "" and int(fdi_digit) != -1:
                    # Double check it is in 1~6 range just in case
                    if 1 <= int(fdi_digit) <= 6:
                        self.rows.append(row)

        self.dataset_dir = Path(dataset_dir)
        self.jaw = jaw
        self.split = split
        self.transform = transform
        self.crop_cache = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_name = row['image_name']
        fdi_digit = int(row['fdi_last_digit'])
        fdi_number = int(row['fdi_number'])

        # Target label is 0-indexed (0 to 5 range for 6 classes: fdi_digit 1 to 6)
        target_label = fdi_digit - 1

        cache_key = (image_name, fdi_number)

        # 1. Load original image (always re-read; only the bbox is cached, not the pixels)
        img_path = self.dataset_dir / self.split / "images" / self.jaw / image_name
        image = cv2.imread(str(img_path))
        if image is None:
            # Fallback if image file is not found
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        h, w = image.shape[:2]

        if cache_key in self.crop_cache:
            x1, y1, x2, y2 = self.crop_cache[cache_key]
        else:
            x1, y1, x2, y2 = 0, 0, w, h

            # 2. Look up this tooth's GT polygon by fdi_number for an accurate crop
            image_name_no_ext = os.path.splitext(image_name)[0]
            json_path = self.dataset_dir / self.split / "labels_json" / self.jaw / f"{image_name_no_ext}.json"

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

                                # Add some crop padding
                                pad = 10
                                x1 = int(max(0, x_min - pad))
                                y1 = int(max(0, y_min - pad))
                                x2 = int(min(w, x_max + pad))
                                y2 = int(min(h, y_max + pad))
                                break
                except Exception as e:
                    print(f"Error loading GT json for crop: {e}")

            self.crop_cache[cache_key] = (x1, y1, x2, y2)

        # 3. Crop the tooth
        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            cropped = np.zeros((224, 224, 3), dtype=np.uint8)

        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cropped_rgb)
            
        if self.transform:
            img_tensor = self.transform(pil_img)
        else:
            img_tensor = transforms.ToTensor()(pil_img)
            
        # 4. Meta features: [x1, y1, x2, y2, theta]
        meta = torch.tensor([
            float(row['x1']),
            float(row['y1']),
            float(row['x2']),
            float(row['y2']),
            float(row['theta'])
        ], dtype=torch.float32)
        
        return img_tensor, meta, target_label

class ToothPositionClassifier(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(ToothPositionClassifier, self).__init__()
        # ImageNet-pretrained backbone for training from scratch. Inference-only callers (e.g.
        # server.py, which immediately overwrites every weight via load_state_dict) should pass
        # pretrained=False to skip this fetch entirely - it's wasted network/disk I/O when the
        # weights are about to be discarded anyway.
        if not pretrained:
            self.resnet = models.resnet18(weights=None)
        else:
            try:
                self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                print("Pretrained ResNet18 weights loaded.")
            except (ImportError, AttributeError):
                self.resnet = models.resnet18(pretrained=True)
                print("Pretrained ResNet18 weights loaded (legacy).")

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove classifier head
        
        # Meta-feature MLP
        self.meta_fc = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU()
        )
        
        # Combined classifier head (now 6 output classes by default)
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, img, meta):
        img_features = self.resnet(img)
        meta_features = self.meta_fc(meta)
        combined = torch.cat((img_features, meta_features), dim=1)
        logits = self.classifier(combined)
        return torch.softmax(logits, dim=1)

def train_model(jaw, dataset_dir, train_csv_path, val_csv_path, model_save_path, runs_dir, epochs, batch_size, lr, patience=10):
    # Ensure save directory and runs directory exist
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(runs_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Image transforms
    # No RandomHorizontalFlip/RandomRotation: the [x1,y1,x2,y2,theta] meta vector is read
    # straight from the CSV regardless of how the image is augmented (see
    # ToothDataset.__getitem__), so a flipped/rotated crop paired with unchanged geometry
    # would train on a mismatched pair. ColorJitter is purely photometric, so it's unaffected.
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
    
    # 2. Datasets & Loaders
    print(f"Loading train dataset from: {train_csv_path}")
    train_dataset = ToothDataset(csv_path=train_csv_path, dataset_dir=dataset_dir, jaw=jaw, split="train", transform=train_transform)
    
    print(f"Loading validation dataset from: {val_csv_path}")
    val_dataset = ToothDataset(csv_path=val_csv_path, dataset_dir=dataset_dir, jaw=jaw, split="val", transform=val_transform)
    
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
    
    # 3. Model setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Predict 6 classes
    model = ToothPositionClassifier(num_classes=6)
    model = model.to(device)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_acc = 0.0
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    # 4. Training loop
    print(f"\nStarting ResNet18 Tooth Classifier (6 classes) training for {jaw.upper()} jaw...")
    for epoch in range(epochs):
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
        
        # Validation phase
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
              
        # Save-best and early-stopping both track validation accuracy - the metric the saved
        # checkpoint is actually judged on - so the two log lines below can never contradict
        # each other the way they could when one tracked accuracy and the other tracked loss.
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
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

    # 5. Plot and save metrics history plot
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss History ({jaw.upper()} Model)')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accs, label='Train Acc', marker='o')
    plt.plot(epochs_range, val_accs, label='Val Acc', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy History ({jaw.upper()} Model)')
    plt.legend()
    
    # F1-Score plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_f1s, label='Train F1', marker='o')
    plt.plot(epochs_range, val_f1s, label='Val F1', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score (Macro)')
    plt.title(f'F1-Score History ({jaw.upper()} Model)')
    plt.legend()
    
    plt.tight_layout()
    plot_path = Path(runs_dir) / f"{jaw}_metrics_history.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved metric plots to {plot_path.resolve()}")
    
    # 6. Generate final confusion matrix using the best model weights
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
        plt.title(f"Confusion Matrix (Best {jaw.upper()} Model)")
        plt.tight_layout()
        
        cm_path = Path(runs_dir) / f"{jaw}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion matrix plot saved at: {cm_path.resolve()}")

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_dir = project_root.parent / "dataset"
    csv_dir = project_root / "ResNet" / "tooth" / "csv"
    model_dir = project_root / "ResNet" / "tooth" / "model"
    runs_dir = project_root / "ResNet" / "tooth" / "runs"
    
    parser = argparse.ArgumentParser(description="Train ResNet18 tooth classifier model using metadata fusion.")
    parser.add_argument("--jaw", type=str, default="lower", choices=("lower", "upper"),
                        help="Jaw model to train: lower or upper (default: lower)")
    parser.add_argument("--dataset_dir", type=str, default=str(dataset_dir),
                        help="Path to the dataset folder")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of epochs to train (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training (default: 128)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience in epochs (default: 10)")

    args = parser.parse_args()
    
    train_csv_path = csv_dir / f"{args.jaw}_features_train.csv"
    val_csv_path = csv_dir / f"{args.jaw}_features_val.csv"
    model_save_path = model_dir / f"{args.jaw}_best.pth"
    
    train_model(
        jaw=args.jaw,
        dataset_dir=args.dataset_dir,
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        model_save_path=model_save_path,
        runs_dir=runs_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience
    )

if __name__ == "__main__":
    main()
