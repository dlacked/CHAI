"""
Evaluates the full pipeline's two models on the held-out test split
(dataset/test/, never touched by training or validation):

  1. Tooth recognition - is a segmented region actually a tooth? (accuracy/
     precision/recall from IoU-matching YOLO's predicted boxes against GT)
  2. Tooth number - is the predicted FDI last digit the true one? (accuracy/
     precision/recall from the ResNet+ViT pipeline's final prediction)

Reuses the project's existing pipelines instead of reimplementing them:
functions/features/main.py (GT feature CSV extraction) and
ViT/arch/build_dataset.py (ResNet crop/feature/probability extraction per arch).

Usage:
    .venv/Scripts/python.exe functions/graph/evaluate_test_performance.py
"""
import csv as csv_module
import importlib.util
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay,
)

# All 32 FDI tooth numbers (8 per quadrant), in reading order for the per-number bar charts
ALL_FDI_NUMBERS = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28,
                   31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]

GRAPH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GRAPH_DIR.parent.parent          # .../CHAI/CHAI
DATASET_DIR = PROJECT_ROOT.parent / "dataset"   # .../CHAI/dataset
RESULTS_DIR = GRAPH_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_DIR = PROJECT_ROOT / "ResNet" / "tooth" / "csv"
RESNET_MODEL_DIR = PROJECT_ROOT / "ResNet" / "tooth" / "model"
ARCH_MODEL_DIR = PROJECT_ROOT / "ViT" / "arch" / "model"

# dataviz reference palette - fixed categorical order (blue = slot 1, orange = slot 2)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(MUTED)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def annotate_bars(ax, bars):
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=INK)


# ---------------------------------------------------------------------------
# Part 1: Tooth recognition - is a segmented region actually a tooth?
# Matches YOLO's predicted boxes against GT boxes (dataset/test/labels, YOLO
# polygon format) via IoU on the test set, then reduces to a single
# TP/FP/FN confusion so "accuracy" has a well-defined meaning for a detector
# (no true negatives exist for object detection, so accuracy = TP/(TP+FP+FN),
# the same convention IoU-based detection accuracy uses elsewhere).
# ---------------------------------------------------------------------------
def _load_gt_boxes_with_numbers(json_path):
    """Reads dataset/test/labels_json's per-tooth polygons (same GT source functions/features
    and the training pipelines use) and reduces each to a bbox + its FDI number, so recognition
    misses/hits can be attributed to a specific tooth position."""
    boxes, numbers = [], []
    if not json_path.exists():
        return boxes, numbers
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for t in data.get("tooth", []):
        num = t.get("teeth_num")
        seg = t.get("segmentation", [])
        if num is None or not seg:
            continue
        if isinstance(seg[0], list) and len(seg[0]) == 2:
            poly = np.array(seg, dtype=np.float64)
        else:
            poly = np.array(seg, dtype=np.float64).reshape(-1, 2)
        x_min, y_min = poly.min(axis=0)
        x_max, y_max = poly.max(axis=0)
        boxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])
        numbers.append(int(num))
    return boxes, numbers


def _box_iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_tooth_recognition(iou_threshold=0.5, conf=0.25):
    from ultralytics import YOLO

    weights_path = PROJECT_ROOT / "YOLO" / "runs" / "segment" / "weights" / "best.pt"
    if not weights_path.exists():
        weights_path = PROJECT_ROOT / "YOLO" / "yolov8n-seg.pt"
    print(f"Loading YOLO weights from {weights_path}...")
    model = YOLO(str(weights_path))

    tp = fp = fn = 0
    per_number_tp = defaultdict(int)
    per_number_fp = defaultdict(int)
    per_number_fn = defaultdict(int)

    for jaw in ("lower", "upper"):
        img_dir = DATASET_DIR / "test" / "images" / jaw
        json_dir = DATASET_DIR / "test" / "labels_json" / jaw
        n_images = sum(1 for _ in img_dir.glob("*.png"))
        print(f"Running YOLO inference on {n_images} {jaw} test images...")

        # Pass the folder path itself (not a Python list of paths) - ultralytics streams
        # images lazily from disk this way. A list, by contrast, gets eagerly pre-loaded
        # into memory in full via autocast_list() before inference even starts, which
        # blew up with a MemoryError on this many full-resolution images.
        results = model.predict(source=str(img_dir), conf=conf, stream=True, verbose=False)
        for result in results:
            img_path = Path(result.path)
            json_path = json_dir / f"{img_path.stem}.json"
            gt_boxes, gt_numbers = _load_gt_boxes_with_numbers(json_path)

            if result.boxes is None or len(result.boxes) == 0:
                fn += len(gt_boxes)
                for num in gt_numbers:
                    per_number_fn[num] += 1
                continue

            pred_boxes = result.boxes.xyxy.cpu().numpy().tolist()
            confs = result.boxes.conf.cpu().numpy().tolist()
            order = np.argsort(confs)[::-1]

            matched_gt = set()
            unmatched_preds = []
            for idx in order:
                pbox = pred_boxes[idx]
                best_iou, best_j = 0.0, -1
                for j, gbox in enumerate(gt_boxes):
                    if j in matched_gt:
                        continue
                    v = _box_iou(pbox, gbox)
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_iou >= iou_threshold:
                    tp += 1
                    matched_gt.add(best_j)
                    per_number_tp[gt_numbers[best_j]] += 1
                else:
                    fp += 1
                    unmatched_preds.append(pbox)
            for j in range(len(gt_boxes)):
                if j not in matched_gt:
                    fn += 1
                    per_number_fn[gt_numbers[j]] += 1

            # A false positive has no GT of its own, so it can't be attributed to a tooth
            # number directly - instead charge it to whichever GT box (matched or not) it
            # overlaps most, i.e. "this spurious/duplicate detection sits on tooth X". If the
            # image has no GT teeth at all, the FP can't be located to any number and is
            # dropped from the per-number breakdown (still counted in the pooled fp above).
            if gt_boxes:
                for pbox in unmatched_preds:
                    best_iou, best_j = -1.0, -1
                    for j, gbox in enumerate(gt_boxes):
                        v = _box_iou(pbox, gbox)
                        if v > best_iou:
                            best_iou, best_j = v, j
                    per_number_fp[gt_numbers[best_j]] += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # Per-number recall is unambiguous (matched / total GT for that tooth). Per-number
    # precision uses the nearest-GT FP attribution above, since YOLO's single "teeth" class
    # never proposes a number on its own.
    accuracy_by_number, precision_by_number, recall_by_number = {}, {}, {}
    for num in ALL_FDI_NUMBERS:
        n_tp, n_fp, n_fn = per_number_tp[num], per_number_fp[num], per_number_fn[num]
        total = n_tp + n_fn
        if total > 0:
            accuracy_by_number[num] = n_tp / total
            recall_by_number[num] = n_tp / total
        if (n_tp + n_fp) > 0:
            precision_by_number[num] = n_tp / (n_tp + n_fp)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "accuracy_by_number": accuracy_by_number,
        "precision_by_number": precision_by_number,
        "recall_by_number": recall_by_number,
    }


def plot_metric_triplet(metrics, title, out_path):
    labels = ["Accuracy", "Precision", "Recall"]
    values = [metrics["accuracy"], metrics["precision"], metrics["recall"]]

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=SURFACE)
    bars = ax.bar(labels, values, color=BLUE, width=0.5, zorder=3)
    annotate_bars(ax, bars)

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", color=INK)
    ax.set_title(title, color=INK)
    style_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close()


def plot_metrics_by_number(metrics_by_number, title, out_path):
    """Three stacked small multiples (Accuracy / Precision / Recall), one bar per FDI tooth
    number that appeared in the test set, in fixed reading order (quadrant 1 -> 2 -> 3 -> 4)
    so the same position lands in the same spot across the two pipelines' charts. Stacked
    subplots (one axis per metric) rather than 3 bars per number - 32 numbers x 3 grouped bars
    would be too dense to read the annotated values on."""
    metric_names = ["accuracy", "precision", "recall"]
    metric_titles = ["Accuracy", "Precision", "Recall"]

    present_numbers = [n for n in ALL_FDI_NUMBERS if n in metrics_by_number[metric_names[0]]
                        or n in metrics_by_number[metric_names[1]] or n in metrics_by_number[metric_names[2]]]
    labels = [str(n) for n in present_numbers]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), facecolor=SURFACE)
    for ax, name, mtitle in zip(axes, metric_names, metric_titles):
        values = [metrics_by_number[name].get(n, 0.0) for n in present_numbers]
        bars = ax.bar(labels, values, color=BLUE, width=0.6, zorder=3)
        annotate_bars(ax, bars)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(mtitle, color=INK)
        style_axes(ax)
        plt.setp(ax.get_xticklabels(), color=INK)

    axes[-1].set_xlabel("FDI Tooth Number", color=INK)
    fig.suptitle(title, color=INK, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Part 2: Tooth number (FDI last digit) performance - ResNet-only vs ResNet+ViT
# ---------------------------------------------------------------------------
def load_fdi_numbers_per_arch(csv_path, max_teeth_per_arch=12):
    """Reproduces ToothDataset's row filter and build_split's per-image grouping/truncation
    (ViT/arch/build_dataset.py) using the FDI number column that build_split's own return
    value drops, so the result lines up 1:1 with arch_build.build_split's `sequences` list -
    lets per-tooth predictions be attributed back to a specific FDI number."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv_module.DictReader(f):
            fdi_digit = row.get("fdi_last_digit")
            if fdi_digit not in (None, "") and 1 <= int(fdi_digit) <= 6:
                rows.append(row)

    groups = OrderedDict()
    for row in rows:
        groups.setdefault(row["image_name"], []).append(row)

    fdi_lists = []
    for group_rows in groups.values():
        if len(group_rows) > max_teeth_per_arch:
            group_rows = group_rows[:max_teeth_per_arch]
        fdi_lists.append([int(r["fdi_number"]) for r in group_rows])
    return fdi_lists


def evaluate_tooth_number():
    features_main = load_module("features_main", PROJECT_ROOT / "functions" / "features" / "main.py")
    arch_build = load_module("arch_build_dataset", PROJECT_ROOT / "ViT" / "arch" / "build_dataset.py")
    ArchToothTransformer = load_module(
        "vit_arch_model_eval", PROJECT_ROOT / "ViT" / "arch" / "model.py"
    ).ArchToothTransformer

    per_jaw = {}
    for jaw in ("lower", "upper"):
        csv_path = CSV_DIR / f"{jaw}_features_test.csv"
        if not csv_path.exists():
            print(f"\nExtracting GT features for {jaw}/test...")
            features_main.process_jaw(jaw, "test", str(DATASET_DIR), csv_path)

        print(f"\nBuilding {jaw}/test arch sequences (ResNet forward pass)...")
        sequences = arch_build.build_split(jaw, "test", str(DATASET_DIR), str(CSV_DIR), str(RESNET_MODEL_DIR), device)
        fdi_number_lists = load_fdi_numbers_per_arch(csv_path)

        arch_weights = ARCH_MODEL_DIR / f"{jaw}_best.pth"
        arch_model = None
        if arch_weights.exists():
            print(f"Loading ViT arch transformer for {jaw}...")
            arch_model = ArchToothTransformer(num_classes=6)
            arch_model.load_state_dict(torch.load(arch_weights, map_location=device))
            arch_model.to(device)
            arch_model.eval()
        else:
            print(f"No ViT arch transformer for {jaw} - refined == ResNet-only.")

        y_true, y_resnet, y_refined, y_fdi_number, y_image_name = [], [], [], [], []
        with torch.no_grad():
            for seq, fdi_numbers in zip(sequences, fdi_number_lists):
                target = seq["target"]
                prob_vec = seq["prob_vec"]
                geom = seq["geom"]
                img_vec = seq["img_vec"]

                y_true.extend(target.tolist())
                y_fdi_number.extend(fdi_numbers)
                y_image_name.extend([seq["image_name"]] * len(target))
                y_resnet.extend(torch.argmax(prob_vec, dim=1).tolist())

                if arch_model is not None:
                    k = geom.size(0)
                    key_padding_mask = torch.zeros(1, k, dtype=torch.bool, device=device)
                    refined_logits = arch_model(
                        geom.unsqueeze(0).to(device),
                        img_vec.unsqueeze(0).to(device),
                        prob_vec.unsqueeze(0).to(device),
                        key_padding_mask
                    )
                    y_refined.extend(torch.argmax(refined_logits[0], dim=1).cpu().tolist())
                else:
                    y_refined.extend(torch.argmax(prob_vec, dim=1).tolist())

        per_jaw[jaw] = {
            "y_true": y_true, "y_resnet": y_resnet, "y_refined": y_refined,
            "y_fdi_number": y_fdi_number, "y_image_name": y_image_name,
            "n_arches": len(sequences), "n_teeth": len(y_true),
            "has_arch_model": arch_model is not None,
        }
        print(f"{jaw}: {len(sequences)} arches, {len(y_true)} teeth.")

    return per_jaw


def plot_tooth_number_confusion(per_jaw, pred_key, label, out_path_template):
    """Plots one confusion matrix per jaw for a single prediction source (pred_key is
    'y_resnet' for the ResNet-only baseline or 'y_refined' for the pipeline's final output -
    both are already collected per-tooth in evaluate_tooth_number(), just not both plotted
    before). Called twice from main() so the two are directly comparable side by side."""
    display_labels = ["1", "2", "3", "4", "5", "6"]
    for jaw, data in per_jaw.items():
        cm = confusion_matrix(data["y_true"], data[pred_key], labels=list(range(6)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        fig, ax = plt.subplots(figsize=(6, 6), facecolor=SURFACE)
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
        ax.set_title(f"Tooth Number Confusion Matrix - {jaw.capitalize()} Jaw ({label})", color=INK)
        plt.tight_layout()
        out_path = Path(str(out_path_template).format(jaw=jaw))
        plt.savefig(out_path, dpi=150, facecolor=SURFACE)
        plt.close()


def compute_tooth_number_metrics(per_jaw, pred_key):
    """Accuracy/precision/recall (macro, over the 6 last-digit classes) of one prediction
    source (pred_key: 'y_resnet' for the ResNet-only baseline, 'y_refined' for the pipeline's
    final ResNet+ViT output) against the GT last digit, pooled across both jaws."""
    all_true, all_pred = [], []
    for jaw in ("lower", "upper"):
        d = per_jaw[jaw]
        all_true += d["y_true"]
        all_pred += d[pred_key]

    return {
        "accuracy": accuracy_score(all_true, all_pred),
        "precision": precision_score(all_true, all_pred, average="macro", zero_division=0),
        "recall": recall_score(all_true, all_pred, average="macro", zero_division=0),
    }


def compute_tooth_number_metrics_by_number(per_jaw):
    """Accuracy/precision/recall of the final (ResNet+ViT where available) prediction, grouped
    by the tooth's true full FDI number rather than just its last digit - e.g. is 48 (wisdom
    tooth) harder to get right than 41? Reconstructs a full predicted FDI number as
    known_tens*10 + predicted_digit (tens comes from the tooth's own geometry, exactly like the
    live site's computeQuadrantTens - the model only ever predicts the last digit), so this is
    a faithful multiclass precision/recall over the tooth positions the model can predict
    (digits 1-6; the 6-class model was never trained on the 7/8 wisdom-tooth digits, so those
    FDI numbers won't appear here even though the segmentation task above covers all 32)."""
    all_true_fdi, all_pred_fdi = [], []
    for jaw in ("lower", "upper"):
        d = per_jaw[jaw]
        for pred_digit, fdi_number in zip(d["y_refined"], d["y_fdi_number"]):
            tens = fdi_number // 10
            all_true_fdi.append(fdi_number)
            all_pred_fdi.append(tens * 10 + (pred_digit + 1))

    present = sorted(set(all_true_fdi) | set(all_pred_fdi))
    precision, recall, _, _ = precision_recall_fscore_support(
        all_true_fdi, all_pred_fdi, labels=present, average=None, zero_division=0
    )

    # Per-class recall == per-class hit-rate here (single predicted label per instance), so
    # "accuracy" and "recall" coincide - kept as separate keys for symmetry with the
    # segmentation side, where they're genuinely different things.
    accuracy_by_number = {num: float(r) for num, r in zip(present, recall) if num in ALL_FDI_NUMBERS}
    precision_by_number = {num: float(p) for num, p in zip(present, precision) if num in ALL_FDI_NUMBERS}
    recall_by_number = dict(accuracy_by_number)

    return {
        "accuracy": accuracy_by_number,
        "precision": precision_by_number,
        "recall": recall_by_number,
    }


def main():
    summary = {}

    print("=" * 70)
    print("Part 1: Tooth recognition - is a segmented region actually a tooth?")
    print("=" * 70)
    recognition_metrics = evaluate_tooth_recognition()
    plot_metric_triplet(
        recognition_metrics,
        "Tooth Recognition Performance (Test Set)",
        RESULTS_DIR / "tooth_recognition_metrics.png"
    )
    plot_metrics_by_number(
        {
            "accuracy": recognition_metrics["accuracy_by_number"],
            "precision": recognition_metrics["precision_by_number"],
            "recall": recognition_metrics["recall_by_number"],
        },
        "Tooth Recognition by Tooth Number (Segmentation, Test Set)",
        RESULTS_DIR / "tooth_recognition_by_number.png"
    )
    summary["tooth_recognition"] = recognition_metrics

    print("\n" + "=" * 70)
    print("Part 2: Tooth number - is the predicted FDI digit the true one?")
    print("=" * 70)
    per_jaw = evaluate_tooth_number()

    # ResNet-only baseline vs the pipeline's final ResNet+ViT output, plotted and scored
    # separately so the refinement's effect is directly visible (both predictions were already
    # collected per-tooth in evaluate_tooth_number(), just not both surfaced before).
    plot_tooth_number_confusion(
        per_jaw, "y_resnet", "ResNet-only", RESULTS_DIR / "tooth_number_confusion_resnet_{jaw}.png"
    )
    plot_tooth_number_confusion(
        per_jaw, "y_refined", "ResNet+ViT", RESULTS_DIR / "tooth_number_confusion_vit_{jaw}.png"
    )

    resnet_metrics = compute_tooth_number_metrics(per_jaw, "y_resnet")
    vit_metrics = compute_tooth_number_metrics(per_jaw, "y_refined")
    plot_metric_triplet(
        resnet_metrics,
        "Tooth Number Performance - ResNet-only (Test Set)",
        RESULTS_DIR / "tooth_number_metrics_resnet.png"
    )
    plot_metric_triplet(
        vit_metrics,
        "Tooth Number Performance - ResNet+ViT (Test Set)",
        RESULTS_DIR / "tooth_number_metrics_vit.png"
    )

    number_metrics_by_number = compute_tooth_number_metrics_by_number(per_jaw)
    plot_metrics_by_number(
        number_metrics_by_number,
        "Tooth Number by Tooth Number (ResNet+ViT, Test Set)",
        RESULTS_DIR / "tooth_number_by_number.png"
    )
    summary["tooth_number"] = {
        "metrics": {"resnet_only": resnet_metrics, "resnet_vit": vit_metrics},
        "metrics_by_number": number_metrics_by_number,
        "per_jaw": {
            jaw: {k: v for k, v in d.items() if k not in ("y_true", "y_resnet", "y_refined", "y_fdi_number", "y_image_name")}
            for jaw, d in per_jaw.items()
        },
    }

    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("Done. Results saved to", RESULTS_DIR.resolve())
    print("=" * 70)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
