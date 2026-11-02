"""
Full-test-set Exact-Match Accuracy for a CHAI model variant (backbone ablation or the
no-ImageNet-pretraining ablation), for Supplementary Table S6. Reuses evaluate_chai.py's exact
crop/inference loop (same int()-truncated pad=10 box, same features_test.csv meta-vector) so
numbers are directly comparable to the published "Proposed (Ours)" 0.9976 full-test Exact-Match
Accuracy, just swapping in a different checkpoint/architecture.

Usage:
    .venv/Scripts/python.exe ResNet/evaluate_chai_variant.py --variant no_pretrain
    .venv/Scripts/python.exe ResNet/evaluate_chai_variant.py --variant resnet34
    .venv/Scripts/python.exe ResNet/evaluate_chai_variant.py --variant resnet50
    .venv/Scripts/python.exe ResNet/evaluate_chai_variant.py --variant efficientnet_b0
    .venv/Scripts/python.exe ResNet/evaluate_chai_variant.py --variant mobilenet_v3_large
    .venv/Scripts/python.exe ResNet/evaluate_chai_variant.py --variant convnext_tiny
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

TOOTH_DIR = Path(__file__).resolve().parent
_bootstrap_root = TOOTH_DIR.parent
if str(_bootstrap_root) not in sys.path:
    sys.path.append(str(_bootstrap_root))
from config import PathConfig
PROJECT_ROOT = PathConfig.ROOT
DATASET_DIR = PROJECT_ROOT.parent / "dataset"
IMAGE_DIR = DATASET_DIR / "test" / "images"
JSON_DIR = DATASET_DIR / "test" / "labels_json"
CSV_PATH = TOOTH_DIR / "csv" / "features_test.csv"

if str(TOOTH_DIR) not in sys.path:
    sys.path.append(str(TOOTH_DIR))
from train_mirrored import ToothPositionClassifierNoTheta  # noqa: E402
from train_chai_backbone import ToothPositionClassifierBackbone  # noqa: E402
from postprocess import resolve_arch_chai  # noqa: E402

JAW_TENS = {"upper": (1, 2), "lower": (3, 4)}

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_rows():
    rows_by_arch = {}
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["jaw"], row["image_name"])
            rows_by_arch.setdefault(key, []).append(row)
    return rows_by_arch


def build_model(variant, device):
    if variant == "no_pretrain":
        model_path = TOOTH_DIR / "model_no_pretrain" / "best.pth"
        model = ToothPositionClassifierNoTheta(num_classes=12, pretrained=False)
    else:
        model_path = TOOTH_DIR / f"model_{variant}" / "best.pth"
        model = ToothPositionClassifierBackbone(variant, num_classes=12, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"Loaded {model_path}")
    return model


def evaluate(variant):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = build_model(variant, device)

    rows_by_arch = load_rows()
    n_arches = len(rows_by_arch)
    n_teeth_total = sum(len(v) for v in rows_by_arch.values())
    print(f"{n_arches} arches, {n_teeth_total} teeth total")

    n_tooth_correct = 0
    n_arch_correct = 0
    n_done = 0
    for (jaw, image_name), rows in rows_by_arch.items():
        stem = Path(image_name).stem
        image_full = cv2.imread(str(IMAGE_DIR / jaw / image_name))
        h, w = image_full.shape[:2]

        with open(JSON_DIR / jaw / f"{stem}.json", "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        box_by_fdi = {}
        for t in gt_data.get("tooth", []):
            num = t.get("teeth_num")
            seg = t.get("segmentation", [])
            if num is None or not seg:
                continue
            poly = np.array(seg) if (isinstance(seg[0], list) and len(seg[0]) == 2) else np.array(seg).reshape(-1, 2)
            x_min, y_min = poly.min(axis=0)
            x_max, y_max = poly.max(axis=0)
            pad = 10
            box_by_fdi[num] = (
                int(max(0, x_min - pad)), int(max(0, y_min - pad)),
                int(min(w, x_max + pad)), int(min(h, y_max + pad)),
            )

        imgs, metas = [], []
        for row in rows:
            x1, y1, x2, y2 = box_by_fdi[int(row["fdi_number"])]
            cropped = image_full[y1:y2, x1:x2]
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(cropped_rgb)
            imgs.append(val_transform(image))
            metas.append([float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])])

        img_batch = torch.stack(imgs).to(device)
        meta_batch = torch.tensor(metas, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = model(img_batch, meta_batch).cpu().numpy()

        class12_preds = resolve_arch_chai(probs)

        arch_correct = True
        for i, row in enumerate(rows):
            class12 = int(class12_preds[i])
            local_quadrant, digit0 = class12 // 6, class12 % 6
            true_digit0 = int(row["fdi_last_digit"]) - 1
            fdi_number = int(row["fdi_number"])
            tens_pred = JAW_TENS[jaw][local_quadrant]
            is_correct = (digit0 == true_digit0) and (tens_pred == fdi_number // 10)
            n_tooth_correct += int(is_correct)
            arch_correct = arch_correct and is_correct
        n_arch_correct += int(arch_correct)

        n_done += 1
        if n_done % 1000 == 0:
            print(f"  {n_done}/{n_arches} arches")

    tooth_acc = n_tooth_correct / n_teeth_total
    arch_acc = n_arch_correct / n_arches
    print(f"\n=== {variant}: full test set ===")
    print(f"Per-tooth Accuracy: {tooth_acc:.4f} ({n_tooth_correct}/{n_teeth_total})")
    print(f"Arch-level Exact-Match Accuracy: {arch_acc:.4f} ({n_arch_correct}/{n_arches})")
    return tooth_acc, arch_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True,
                         choices=["no_pretrain", "resnet18", "resnet34", "resnet50",
                                  "efficientnet_b0", "mobilenet_v3_large", "convnext_tiny"])
    args = parser.parse_args()
    evaluate(args.variant)
