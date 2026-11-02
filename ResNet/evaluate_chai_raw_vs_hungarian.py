"""
Restructures Table 3 (tab:method_comparison, "Effect of Hungarian Assignment on the Unified
Classifier") into the same per-tooth (Precision/Sensitivity/Specificity/Macro F1/Accuracy) +
per-arch (Exact-Match Accuracy only) format already used for Table 1/2, on the missing-tooth
subset only (Full Test Set columns dropped per the same convention).

Reuses evaluate_chai.py's exact inference loop (same crops, same model, same val_transform) but
computes BOTH the raw per-tooth argmax and the Hungarian-resolved class12 from the same probs
matrix per arch, since evaluate_chai.py's saved chai_per_jaw.pkl only kept the Hungarian result -
Table 1/2's "before Hungarian" comparison needs both from the same forward pass. Separate script
per this project's preference for new sibling scripts over editing production evaluation code.

Usage:
    .venv/Scripts/python.exe ResNet/evaluate_chai_raw_vs_hungarian.py
"""
import csv
import importlib.util
import json
import pickle
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
MODEL_PATH = TOOTH_DIR / "model" / "best.pth"
RESULTS_DIR = TOOTH_DIR / "results"

if str(TOOTH_DIR) not in sys.path:
    sys.path.append(str(TOOTH_DIR))
from train_mirrored import ToothPositionClassifierNoTheta  # noqa: E402
from postprocess import resolve_arch_chai  # noqa: E402

JAW_TENS = {"upper": (1, 2), "lower": (3, 4)}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def blank_per_jaw():
    return {
        "lower": {"y_true": [], "y_pred": [], "y_tens_pred": [], "prob": [],
                  "y_fdi_number": [], "y_image_name": []},
        "upper": {"y_true": [], "y_pred": [], "y_tens_pred": [], "prob": [],
                  "y_fdi_number": [], "y_image_name": []},
    }


def record(d, class12, row, probs_i):
    local_quadrant, digit0 = class12 // 6, class12 % 6
    true_digit0 = int(row["fdi_last_digit"]) - 1
    fdi_number = int(row["fdi_number"])
    conf_vec = [0.0] * 6
    conf_vec[digit0] = float(probs_i[class12])
    d["y_true"].append(true_digit0)
    d["y_pred"].append(digit0)
    d["y_tens_pred"].append(JAW_TENS[row["jaw"]][local_quadrant])
    d["prob"].append(conf_vec)
    d["y_fdi_number"].append(fdi_number)
    d["y_image_name"].append(row["image_name"])


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ToothPositionClassifierNoTheta(num_classes=12, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded {MODEL_PATH}")

    rows_by_arch = load_rows()
    print(f"{len(rows_by_arch)} arches, {sum(len(v) for v in rows_by_arch.values())} teeth total")

    per_jaw_raw = blank_per_jaw()
    per_jaw_hungarian = blank_per_jaw()

    n_done = 0
    for (jaw, image_name), rows in rows_by_arch.items():
        for row in rows:
            row["jaw"] = jaw  # record() reads row["jaw"]
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

        raw_preds = probs.argmax(axis=1)
        hungarian_preds = resolve_arch_chai(probs)

        for i, row in enumerate(rows):
            record(per_jaw_raw[jaw], int(raw_preds[i]), row, probs[i])
            record(per_jaw_hungarian[jaw], int(hungarian_preds[i]), row, probs[i])

        n_done += 1
        if n_done % 500 == 0:
            print(f"  {n_done}/{len(rows_by_arch)} arches")

    for jaw in ("lower", "upper"):
        per_jaw_raw[jaw]["n_teeth"] = len(per_jaw_raw[jaw]["y_true"])
        per_jaw_hungarian[jaw]["n_teeth"] = len(per_jaw_hungarian[jaw]["y_true"])

    return per_jaw_raw, per_jaw_hungarian


def main():
    per_jaw_raw, per_jaw_hungarian = evaluate()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "chai_raw_vs_hungarian_per_jaw.pkl", "wb") as f:
        pickle.dump({"raw": per_jaw_raw, "hungarian": per_jaw_hungarian}, f)
    print(f"Saved to {RESULTS_DIR / 'chai_raw_vs_hungarian_per_jaw.pkl'}")

    tooth_module = load_module("chai_tooth_graph_hungarian_ablation", PROJECT_ROOT / "functions" / "eval_metrics.py")

    missing_buckets = tooth_module.build_missing_count_buckets(per_jaw_hungarian)
    missing_bucket_2way = {k: ("missing" if v >= 1 else "complete") for k, v in missing_buckets.items()}

    results = {}
    for label, per_jaw in (("raw", per_jaw_raw), ("hungarian", per_jaw_hungarian)):
        tooth_metrics = tooth_module.compute_tooth_level_metrics_by_bucket(
            per_jaw, missing_bucket_2way, pred_key="y_pred", tens_pred_key="y_tens_pred"
        )["missing"]
        arch_metrics = tooth_module.compute_missing_vs_complete_metrics(
            per_jaw, pred_key="y_pred", prob_field="prob", tens_pred_key="y_tens_pred"
        )["missing"]
        results[label] = {
            "per_tooth": {k: tooth_metrics[k] for k in
                          ("n_teeth", "precision", "recall", "specificity", "f1", "accuracy")},
            "per_arch": {"n_arches": arch_metrics["n_arches"], "accuracy": arch_metrics["accuracy"]},
        }
        print(f"\n=== {label} (missing-tooth subset) ===")
        print("Per-tooth:", results[label]["per_tooth"])
        print("Per-arch:", results[label]["per_arch"])

    with open(RESULTS_DIR / "chai_raw_vs_hungarian_table3.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'chai_raw_vs_hungarian_table3.json'}")


if __name__ == "__main__":
    main()
