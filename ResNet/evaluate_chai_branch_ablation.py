"""
Evaluation for CHAI's branch-ablation variants (image_only, meta_only - see
train_chai_branch_ablation.py) on the FULL test set. Builds a per_jaw dict in the exact same
{jaw: {y_true, y_pred, y_tens_pred, prob, y_fdi_number, y_image_name, n_teeth}} shape
evaluate_chai.py produces (GT-crop-conditioned, no live detection - same convention as every other
CHAI numbering figure in the paper), then reports:
  - per-tooth: Precision/Sensitivity/Specificity/F1 (macro, 6 last-digit classes, gated by tens),
    plus LD/TD/overall Accuracy
  - per-arch: Exact-Match Accuracy

Intended as both a quick interim check (while training is still in progress - just re-run after
each new checkpoint) and, once both variants finish training, the source of the final branch-
ablation table's "Image-only" / "Coords-only" rows.

Usage:
    .venv/Scripts/python.exe ResNet/evaluate_chai_branch_ablation.py --variant image_only
    .venv/Scripts/python.exe ResNet/evaluate_chai_branch_ablation.py --variant meta_only
"""
import sys
import csv
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

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
from train_chai_branch_ablation import build_model  # noqa: E402

import json
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


def evaluate(variant):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = TOOTH_DIR / f"model_{variant}" / "best.pth"
    model = build_model(variant, num_classes=12, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded {model_path}")

    rows_by_arch = load_rows()
    print(f"{len(rows_by_arch)} arches, {sum(len(v) for v in rows_by_arch.values())} teeth total")

    per_jaw = {
        "lower": {"y_true": [], "y_pred": [], "y_tens_pred": [], "prob": [],
                   "y_fdi_number": [], "y_image_name": []},
        "upper": {"y_true": [], "y_pred": [], "y_tens_pred": [], "prob": [],
                   "y_fdi_number": [], "y_image_name": []},
    }

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

        class12_preds = probs.argmax(axis=1)

        d = per_jaw[jaw]
        for i, row in enumerate(rows):
            class12 = int(class12_preds[i])
            local_quadrant, digit0 = class12 // 6, class12 % 6
            true_digit0 = int(row["fdi_last_digit"]) - 1
            fdi_number = int(row["fdi_number"])

            conf_vec = [0.0] * 6
            conf_vec[digit0] = float(probs[i, class12])

            d["y_true"].append(true_digit0)
            d["y_pred"].append(digit0)
            d["y_tens_pred"].append(JAW_TENS[jaw][local_quadrant])
            d["prob"].append(conf_vec)
            d["y_fdi_number"].append(fdi_number)
            d["y_image_name"].append(image_name)

        n_done += 1
        if n_done % 500 == 0:
            print(f"  {n_done}/{len(rows_by_arch)} arches")

    for jaw in ("lower", "upper"):
        per_jaw[jaw]["n_teeth"] = len(per_jaw[jaw]["y_true"])
        print(f"{jaw}: {per_jaw[jaw]['n_teeth']} teeth scored.")

    return per_jaw


WRONG_QUADRANT = 6


def gate_by_tens(y_pred, y_tens_pred, y_fdi_number):
    return [p if tp == (n // 10) else WRONG_QUADRANT
            for p, tp, n in zip(y_pred, y_tens_pred, y_fdi_number)]


def multiclass_specificity(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    specs = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        specs.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return float(np.mean(specs))


def report(per_jaw, variant):
    all_true_ld, all_pred_ld_gated = [], []
    all_true_td, all_pred_td = [], []
    all_full_correct = []
    arch_exact = {}

    for jaw in ("lower", "upper"):
        d = per_jaw[jaw]
        gated = gate_by_tens(d["y_pred"], d["y_tens_pred"], d["y_fdi_number"])
        all_true_ld += d["y_true"]
        all_pred_ld_gated += gated

        for i in range(d["n_teeth"]):
            true_tens = d["y_fdi_number"][i] // 10
            pred_tens = d["y_tens_pred"][i]
            all_true_td.append(true_tens)
            all_pred_td.append(pred_tens)

            full_correct = (pred_tens == true_tens) and (d["y_pred"][i] == d["y_true"][i])
            all_full_correct.append(full_correct)

            arch_key = (jaw, d["y_image_name"][i])
            arch_exact[arch_key] = arch_exact.get(arch_key, True) and full_correct

    digit_labels = list(range(6))
    precision = precision_score(all_true_ld, all_pred_ld_gated, labels=digit_labels, average="macro", zero_division=0)
    sensitivity = recall_score(all_true_ld, all_pred_ld_gated, labels=digit_labels, average="macro", zero_division=0)
    f1 = f1_score(all_true_ld, all_pred_ld_gated, labels=digit_labels, average="macro", zero_division=0)
    specificity = multiclass_specificity(all_true_ld, all_pred_ld_gated, labels=digit_labels)
    ld_acc = sum(p == t for p, t in zip(all_pred_ld_gated, all_true_ld) if p != WRONG_QUADRANT) / len(all_true_ld)
    # LD accuracy over ALL teeth (wrong-quadrant counts as wrong LD too, consistent with Table I's convention)
    ld_acc = sum(1 for p, t in zip(all_pred_ld_gated, all_true_ld) if p == t) / len(all_true_ld)
    td_acc = sum(1 for p, t in zip(all_pred_td, all_true_td) if p == t) / len(all_true_td)
    overall_acc = sum(all_full_correct) / len(all_full_correct)

    n_arches = len(arch_exact)
    arch_exact_match_acc = sum(arch_exact.values()) / n_arches

    print("\n" + "=" * 70)
    print(f"Branch ablation - variant: {variant} (full test set)")
    print("=" * 70)
    print(f"Per-tooth  n={len(all_true_ld)}")
    print(f"  Precision={precision:.4f}  Sensitivity={sensitivity:.4f}  Specificity={specificity:.4f}  F1={f1:.4f}")
    print(f"  LD Acc={ld_acc:.4f}  TD Acc={td_acc:.4f}  Overall Acc={overall_acc:.4f}")
    print(f"Per-arch   n={n_arches}")
    print(f"  Exact-Match Accuracy={arch_exact_match_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a CHAI branch-ablation variant on the full test set.")
    parser.add_argument("--variant", type=str, required=True, choices=["image_only", "meta_only"])
    args = parser.parse_args()

    per_jaw = evaluate(args.variant)
    report(per_jaw, args.variant)


if __name__ == "__main__":
    main()
