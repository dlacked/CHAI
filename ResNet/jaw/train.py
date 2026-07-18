import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

# Set matplotlib backend to Agg to prevent GUI display errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root directory to sys.path to import config.py
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import PathConfig

def train_classifier():
    # Ensure all directories exist
    PathConfig.create_dir()
    
    # Define dataset paths
    train_dir = PathConfig.DATASET_DIR / "train" / "images"
    val_dir = PathConfig.DATASET_DIR / "val" / "images"
    
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Dataset paths do not exist. Please check config.py and ensure the dataset folder structure is correct.\n"
            f"Expected train_dir: {train_dir}\n"
            f"Expected val_dir: {val_dir}"
        )

    # 1. Transforms setup (224x224 input size for ResNet)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Datasets & Loaders
    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transform)
    
    # Print dataset details
    print(f"Classes found: {train_dataset.classes} (Mapping: {train_dataset.class_to_idx})")
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    # 3. Model initialization (ResNet18)
    print("Initializing ResNet18 model...")
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        print("Pretrained ResNet18 model loaded successfully.")
    except (ImportError, AttributeError):
        from torchvision.models import resnet18
        model = resnet18(pretrained=True)
        print("Pretrained ResNet18 model loaded (legacy API).")
        
    # Replace final FC layer for binary classification (2 classes: lower, upper)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    # 4. Training configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 1000
    patience = 10
    best_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Lists to store metrics for plotting
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []
    
    # 5. Training loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        train_preds, train_labels_list = [], []
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            train_preds.extend(preds.cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())
            
            print(f"  [Train] Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f} | Running Acc: {running_corrects.double() / len(train_labels_list):.4f}")
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        epoch_f1 = f1_score(train_labels_list, train_preds, average='macro')
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())
        train_f1s.append(epoch_f1)
        
        # Evaluation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        val_preds, val_labels_list = [], []
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
                
                print(f"  [Val] Batch [{batch_idx+1}/{len(val_loader)}] | Loss: {loss.item():.4f} | Running Acc: {val_corrects.double() / len(val_labels_list):.4f}")
                
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)
        val_epoch_f1 = f1_score(val_labels_list, val_preds, average='macro')
        
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc.item())
        val_f1s.append(val_epoch_f1)
        
        print(f"\nEpoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f} F1: {val_epoch_f1:.4f}")
        
        # Save the best model
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), PathConfig.RESNET_WEIGHTS)
            print(f"  --> Saved new best model weights with validation accuracy: {best_acc:.4f}")

        # Early stopping based on validation loss
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  --> No improvement in Val Loss for {epochs_no_improve}/{patience} epochs.")
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1} (no Val Loss improvement for {patience} epochs).")
                break

    print(f"\nTraining completed. Best validation accuracy: {best_acc:.4f}")

    # 6. Save training metrics history plot
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 5))
    
    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accs, label='Train Acc', marker='o')
    plt.plot(epochs_range, val_accs, label='Val Acc', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy History')
    plt.legend()
    
    # Plot F1-Score
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_f1s, label='Train F1-Score', marker='o')
    plt.plot(epochs_range, val_f1s, label='Val F1-Score', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score (Macro)')
    plt.title('F1-Score History')
    plt.legend()
    
    plt.tight_layout()
    metrics_path = PathConfig.RESNET_RUNS_DIR / "metrics_history.png"
    plt.savefig(metrics_path)
    plt.close()
    print(f"Metrics history plot saved at: {metrics_path}")
    
    # 7. Generate final confusion matrix using the best model weights
    print("\nGenerating final confusion matrix using best weights...")
    if PathConfig.RESNET_WEIGHTS.exists():
        model.load_state_dict(torch.load(PathConfig.RESNET_WEIGHTS, map_location=device))
        model.eval()
        
        final_val_preds, final_val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                final_val_preds.extend(preds.cpu().numpy())
                final_val_labels.extend(labels.numpy())
        
        cm = confusion_matrix(final_val_labels, final_val_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_dataset.classes)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title("Confusion Matrix (Best Model)")
        plt.tight_layout()
        
        cm_path = PathConfig.RESNET_RUNS_DIR / "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion matrix plot saved at: {cm_path}")
        
    print(f"Weights saved at: {PathConfig.RESNET_WEIGHTS}")

if __name__ == "__main__":
    train_classifier()
