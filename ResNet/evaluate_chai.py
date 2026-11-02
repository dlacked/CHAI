"""
Evaluation for the proposed model (CHAI) itself, producing a per_jaw dict in the exact same
{jaw: {y_true, y_pred, y_tens_pred, prob, y_fdi_number, y_image_name, n_teeth}} shape
comparison/ghorbani/evaluate.py and comparison/yoon/evaluate.py build, so the same
functions/eval_metrics.py metric functions can score all three models identically.

Unlike the two comparison reproductions, this does NOT re-run tooth detection - CHAI's own
segmentation accuracy is evaluated separately (Methods, End-to-End Pipeline Latency subsection's
mAP figure), decoupled from the numbering evaluation here, which conditions on the ground-truth
crop for every real tooth (matching how ResNet/csv/features_test.csv - built from GT
segmentation polygons - was already used for every other CHAI numbering figure in the paper).

Crops with the exact same int()-truncated pad=10 box train_chai.py's ToothDatasetChai uses (NOT
the pre-cropped ResNet/cropped_tooth/ cache, which was cropped with PIL's float-coordinate
.crop() - a first version of this script used that cache for speed, but its sub-pixel difference
from the int()-truncated crop shifted a handful of arches' predictions and no longer reproduced
the already-published, McNemar-tested Table 1/2 accuracy figures exactly; re-deriving crops from
the full arch image, one decode per arch, both matches production bit-for-bit and is still far
cheaper than a per-tooth redecode). The [x1,y1,x2,y2] meta-vector itself is read straight from
csv/features_test.csv (already computed by functions/features/coords_chai.py).

Usage:
    .venv/Scripts/python.exe ResNet/evaluate_chai.py
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

# This jaw's two tens values, in local-index order (0, 1) - matches functions/features/main_chai.py.
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

        class12_preds = resolve_arch_chai(probs)

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


def main():
    per_jaw = evaluate()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "chai_per_jaw.pkl", "wb") as f:
        pickle.dump(per_jaw, f)
    print(f"Saved raw per-tooth predictions to {RESULTS_DIR / 'chai_per_jaw.pkl'}")

    tooth_module = load_module("chai_tooth_graph_self", PROJECT_ROOT / "functions" / "eval_metrics.py")

    print("\n" + "=" * 70)
    print("Sanity check against published numbers (test set) - Proposed (Ours)")
    print("=" * 70)
    missing = tooth_module.compute_missing_teeth_metrics(
        per_jaw, pred_key="y_pred", prob_field="prob", tens_pred_key="y_tens_pred"
    )
    for missing_count, m in sorted(missing.items()):
        print(f"missing={missing_count:2d}  n_arches={m['n_arches']:5d}  "
              f"precision={m['precision']:.4f}  recall={m['recall']:.4f}  f1={m['f1']:.4f}  "
              f"acc={m['accuracy']:.4f}")

    complexity_build = load_module(
        "vit_complexity_build_eval_chai", PROJECT_ROOT / "functions" / "features" / "complexity_labels.py")
    complexity_by_image = complexity_build.load_complexity_labels(DATASET_DIR, "test")
    arch_by_complexity = tooth_module.compute_arch_metrics_by_complexity(
        per_jaw, complexity_by_image, pred_key="y_pred", prob_field="prob", tens_pred_key="y_tens_pred"
    )
    complexity_labels = {1: "Class I", 2: "Class II", 3: "Class III"}
    for c, m in sorted(arch_by_complexity.items()):
        print(f"{complexity_labels.get(c, c):10s} n_arches={m['n_arches']:5d}  "
              f"precision={m['precision']:.4f}  recall={m['recall']:.4f}  f1={m['f1']:.4f}  "
              f"acc={m['accuracy']:.4f}")


if __name__ == "__main__":
    main()
