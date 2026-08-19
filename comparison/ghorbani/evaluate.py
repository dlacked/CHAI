"""
Evaluation for the Ghorbani et al. reproduction: chains Stage 1 (detect) + Stage 2 (classify) and
adds the paper's own duplicate-number post-process (Fig. 1's "Duplicate label detected? -> Label
modification based on position" step) before scoring against CHAI's GT.

The paper doesn't give an exact algorithm for that step beyond "reassessed those teeth together
based on their position and assigned new numbers as needed". Modeled here as: split each jaw
image's detections into two geometric halves (left/right of that image's own tooth-cluster
x-midline - a stand-in for "half of the jaw"), infer which quadrant each half is via majority
vote of the classifier's own top-1 predictions, then resolve any repeated last-digit within a
half using ResNet/tooth/postprocess.py's resolve_quadrant_greedy: confidence-sorted-descending,
each tooth claims its highest-ranked still-available digit. That function already exists as one
of CHAI's own three duplicate-resolution comparison points and matches the paper's description
(confidence-first claim, fallback to next-best available number) closely enough to reuse as-is
rather than reimplement.

Two sets of metrics are reported:
  1. Ghorbani's own Table 1 definition - TP/FP/FN via IoU>=0.5 matching between a detection and
     its nearest GT box (TP = detected + correctly numbered, FP = detected + wrong number,
     FN = GT tooth with no matching detection) -> Sensitivity/Precision/F1.
  2. CHAI's arch-level metrics (functions/graph/tooth.py's compute_missing_teeth_metrics /
     compute_missing_vs_complete_metrics / compute_arch_metrics_by_complexity) run on a per_jaw
     dict built in the exact same shape CHAI's own evaluate_tooth_number produces - one entry per
     REAL GT tooth (a missed GT tooth gets a guaranteed-wrong sentinel prediction so it still
     costs accuracy/recall; a spurious detection with no GT match has no slot in this per-real-
     tooth scheme and is reported separately as a diagnostic count instead, consistent with
     Ghorbani's own metric taxonomy above having no category for it either).

Usage:
    .venv/Scripts/python.exe comparison/ghorbani/evaluate.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision import transforms

from ultralytics import YOLO

from train_classify import (
    GHORBANI_DIR, PROJECT_ROOT, DATASET_DIR, MODEL_DIR,
    FDI_CLASSES, FDI_TO_IDX, IMAGE_EXTS, CROP_PAD, IMG_SIZE, poly_bbox, build_model,
)

import sys
sys.path.insert(0, str(PROJECT_ROOT / "ResNet" / "tooth"))
from postprocess import resolve_quadrant_greedy  # noqa: E402

import importlib.util


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RESULTS_DIR = GHORBANI_DIR / "results"
CONF_THRESHOLD = 0.25
IOU_MATCH_THRESHOLD = 0.5
QUADRANTS_BY_JAW = {"upper": (1, 2), "lower": (3, 4)}


def find_latest_detect_weights():
    """detect/ detect-2/ ... are successive train_detect.py attempts (earlier ones failed before
    producing weights during debugging) - pick the newest run directory that actually has a
    weights/best.pt rather than hardcoding a run name."""
    candidates = sorted(
        GHORBANI_DIR.glob("runs/detect*"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    for run_dir in candidates:
        weights = run_dir / "weights" / "best.pt"
        if weights.exists():
            return weights
    raise SystemExit("No trained detect weights found under runs/detect* - run train_detect.py first.")


def iou_matrix(boxes_a, boxes_b):
    """boxes_*: (n, 4) arrays of [x1, y1, x2, y2]. Returns (len(a), len(b)) IoU matrix."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    a = np.asarray(boxes_a)[:, None, :]
    b = np.asarray(boxes_b)[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def match_detections_to_gt(det_boxes, gt_boxes):
    """Hungarian assignment maximizing total IoU, then filtered to IoU>=IOU_MATCH_THRESHOLD.
    Returns (matched_pairs, unmatched_det_idx, unmatched_gt_idx) - matched_pairs is a list of
    (det_idx, gt_idx)."""
    if len(det_boxes) == 0 or len(gt_boxes) == 0:
        return [], list(range(len(det_boxes))), list(range(len(gt_boxes)))
    ious = iou_matrix(det_boxes, gt_boxes)
    row_ind, col_ind = linear_sum_assignment(-ious)
    matched, used_det, used_gt = [], set(), set()
    for r, c in zip(row_ind, col_ind):
        if ious[r, c] >= IOU_MATCH_THRESHOLD:
            matched.append((r, c))
            used_det.add(r)
            used_gt.add(c)
    unmatched_det = [i for i in range(len(det_boxes)) if i not in used_det]
    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in used_gt]
    return matched, unmatched_det, unmatched_gt


def resolve_duplicates_for_image(det_boxes, det_probs, jaw):
    """Ghorbani's Fig.1 duplicate-resolution step, applied to ALL of this image's detections
    (mirrors real deployment, which has no access to GT at inference time). Returns a (n,) array
    of final two-digit FDI class predictions, one per detection, aligned to det_boxes/det_probs.

    det_probs: (n, 24) softmax over FDI_CLASSES. jaw: "upper" or "lower" -> which two quadrants
    (half-jaw halves) are anatomically valid for this image."""
    n = len(det_boxes)
    final_class = np.zeros(n, dtype=int)
    if n == 0:
        return final_class

    valid_quadrants = QUADRANTS_BY_JAW[jaw]
    quadrant_of_class = np.array([c // 10 for c in FDI_CLASSES])
    valid_mask = np.isin(quadrant_of_class, valid_quadrants)

    x_centers = np.array([(b[0] + b[2]) / 2 for b in det_boxes])
    midline = np.median(x_centers)
    side = (x_centers >= midline).astype(int)  # 0 = left half, 1 = right half

    masked_probs = np.where(valid_mask[None, :], det_probs, -np.inf)
    top1_valid_class = np.array(FDI_CLASSES)[masked_probs.argmax(axis=1)]
    top1_valid_quadrant = top1_valid_class // 10

    for s in (0, 1):
        idx = np.where(side == s)[0]
        if len(idx) == 0:
            continue
        votes = top1_valid_quadrant[idx]
        counts = {q: int((votes == q).sum()) for q in valid_quadrants}
        quadrant = max(counts, key=counts.get)
        if counts[quadrant] == 0:
            # No detection in this half voted a valid quadrant at all (all-invalid top-1) -
            # fall back to left-half = lower-numbered quadrant, right-half = higher-numbered,
            # a reasonable default given occlusal photos are laid out roughly symmetrically.
            quadrant = min(valid_quadrants) if s == 0 else max(valid_quadrants)

        digit_class_idx = [FDI_TO_IDX[quadrant * 10 + d] for d in range(1, 7)]
        sub_probs = det_probs[idx][:, digit_class_idx]  # (len(idx), 6)
        digits = resolve_quadrant_greedy(sub_probs, group_keys=[quadrant] * len(idx))
        final_class[idx] = quadrant * 10 + (digits + 1)

    return final_class


def load_gt_teeth(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    teeth = []
    for t in data.get("tooth", []):
        fdi_number = t.get("teeth_num")
        seg = t.get("segmentation", [])
        if fdi_number not in FDI_TO_IDX or not seg:
            continue
        (x_min, y_min), (x_max, y_max) = poly_bbox(seg)
        teeth.append((fdi_number, [float(x_min), float(y_min), float(x_max), float(y_max)]))
    return teeth


def crop_and_classify(image, boxes, classify_model, device, transform):
    if len(boxes) == 0:
        return np.zeros((0, len(FDI_CLASSES)))
    h, w = image.shape[:2]
    crops = []
    for x1, y1, x2, y2 in boxes:
        cx1 = int(max(0, x1 - CROP_PAD))
        cy1 = int(max(0, y1 - CROP_PAD))
        cx2 = int(min(w, x2 + CROP_PAD))
        cy2 = int(min(h, y2 + CROP_PAD))
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            crop = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops.append(transform(Image.fromarray(crop_rgb)))
    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        logits = classify_model(batch)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy()


def evaluate_split(detect_model, classify_model, device, transform, split="test"):
    per_jaw = {}
    ghorbani_totals = {"tp": 0, "fp": 0, "fn": 0, "spurious": 0}

    for jaw in ("lower", "upper"):
        json_dir = DATASET_DIR / split / "labels_json" / jaw
        image_dir = DATASET_DIR / split / "images" / jaw
        if not json_dir.exists():
            continue

        y_true, y_fdi_number, y_image_name = [], [], []
        y_pred, y_tens_pred, prob_field = [], [], []

        json_paths = sorted(json_dir.glob("*.json"))
        print(f"\n{jaw}: evaluating {len(json_paths)} images...")
        for i, json_path in enumerate(json_paths):
            stem = json_path.stem
            image_name = None
            for ext in IMAGE_EXTS:
                if (image_dir / f"{stem}{ext}").exists():
                    image_name = f"{stem}{ext}"
                    break
            if image_name is None:
                continue

            gt_teeth = load_gt_teeth(json_path)
            if not gt_teeth:
                continue
            gt_fdi = [t[0] for t in gt_teeth]
            gt_boxes = [t[1] for t in gt_teeth]

            image = cv2.imread(str(image_dir / image_name))
            if image is None:
                continue

            result = detect_model.predict(image, conf=CONF_THRESHOLD, verbose=False)[0]
            det_boxes = result.boxes.xyxy.cpu().numpy().tolist() if len(result.boxes) else []

            det_probs = crop_and_classify(image, det_boxes, classify_model, device, transform)
            det_final_class = resolve_duplicates_for_image(det_boxes, det_probs, jaw)

            matched, unmatched_det, unmatched_gt = match_detections_to_gt(det_boxes, gt_boxes)
            matched_gt_idx = {gt_i: det_i for det_i, gt_i in matched}

            for gt_i, fdi_number in enumerate(gt_fdi):
                y_true.append(fdi_number % 10 - 1)
                y_fdi_number.append(fdi_number)
                y_image_name.append(image_name)

                if gt_i in matched_gt_idx:
                    det_i = matched_gt_idx[gt_i]
                    pred_class = int(det_final_class[det_i])
                    pred_tens, pred_digit = pred_class // 10, pred_class % 10 - 1
                    y_pred.append(pred_digit)
                    y_tens_pred.append(pred_tens)
                    # 6-slot vector so downstream code (which reads prob_row[pred_digit] as this
                    # tooth's "confidence") can index it the same way as CHAI's own prob_resnet -
                    # only the predicted digit's slot is populated, filled with the RAW
                    # classifier's own probability mass AT the finally-assigned class (not the
                    # raw argmax over all 24 classes, which can differ from pred_class once
                    # quadrant-masking/greedy dedup have moved the assignment away from the
                    # classifier's own top-1 - matches CHAI's prob_resnet[y_postproc] convention:
                    # "how much did the pre-post-process model itself believe in the digit that
                    # ended up being assigned", not "how confident was the model's raw top guess".
                    conf_vec = [0.0] * 6
                    conf_vec[pred_digit] = float(det_probs[det_i][FDI_TO_IDX[pred_class]])
                    prob_field.append(conf_vec)
                    ghorbani_totals["tp" if pred_class == fdi_number else "fp"] += 1
                else:
                    # Missed GT tooth: guaranteed-wrong sentinel (tens=0 never matches a real
                    # 1-4 tens digit) so this still costs accuracy/recall in the arch-level
                    # metrics below, exactly like an undetected tooth should.
                    y_pred.append(0)
                    y_tens_pred.append(0)
                    prob_field.append([0.0] * 6)
                    ghorbani_totals["fn"] += 1

            ghorbani_totals["spurious"] += len(unmatched_det)

            if (i + 1) % 50 == 0 or (i + 1) == len(json_paths):
                print(f"  [{jaw}] {i + 1}/{len(json_paths)} images")

        per_jaw[jaw] = {
            "y_true": y_true, "y_pred": y_pred, "y_tens_pred": y_tens_pred,
            "prob": prob_field, "y_fdi_number": y_fdi_number, "y_image_name": y_image_name,
            "n_teeth": len(y_true),
        }
        print(f"{jaw}: {len(y_true)} GT teeth scored.")

    return per_jaw, ghorbani_totals


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    detect_weights = find_latest_detect_weights()
    print(f"Detect weights: {detect_weights}")
    detect_model = YOLO(str(detect_weights))

    classify_model = build_model(device)
    classify_ckpt = MODEL_DIR / "classify_best.pth"
    if not classify_ckpt.exists():
        raise SystemExit(f"{classify_ckpt} not found - run train_classify.py first.")
    classify_model.load_state_dict(torch.load(classify_ckpt, map_location=device))
    classify_model.eval()
    print(f"Classify weights: {classify_ckpt}")

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    per_jaw, totals = evaluate_split(detect_model, classify_model, device, eval_transform, "test")

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    print("\n" + "=" * 70)
    print("Ghorbani et al. Table 1-style metrics (test set)")
    print("=" * 70)
    print(f"TP={tp}  FP={fp}  FN={fn}  spurious(no-GT-match)={totals['spurious']}")
    print(f"Sensitivity={sensitivity:.4f}  Precision={precision:.4f}  F1={f1:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "ghorbani_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "table1_style": {"tp": tp, "fp": fp, "fn": fn, "spurious": totals["spurious"],
                              "sensitivity": sensitivity, "precision": precision, "f1": f1},
        }, f, indent=2)

    # Reuse CHAI's own arch-level metric functions on this per_jaw (built above in the exact
    # {jaw: {y_true, y_fdi_number, y_image_name, <pred_key>, <tens_pred_key>, <prob_field>}}
    # shape functions/graph/tooth.py's evaluate_tooth_number produces, so no metric logic is
    # duplicated - just point pred_key/tens_pred_key/prob_field at this module's own field names.
    tooth_module = load_module("chai_tooth_graph", PROJECT_ROOT / "functions" / "graph" / "tooth.py")

    print("\n" + "=" * 70)
    print("CHAI-style arch-level metrics (test set) - Ghorbani (YOLOv8n detect+cls)")
    print("=" * 70)
    missing = tooth_module.compute_missing_teeth_metrics(
        per_jaw, pred_key="y_pred", prob_field="prob", tens_pred_key="y_tens_pred"
    )
    for missing_count, m in sorted(missing.items()):
        print(f"missing={missing_count:2d}  n_arches={m['n_arches']:5d}  "
              f"precision={m['precision']:.4f}  recall={m['recall']:.4f}  f1={m['f1']:.4f}  "
              f"acc={m['accuracy']:.4f}  auc={m['auc']:.4f}")

    missing_vs_complete = tooth_module.compute_missing_vs_complete_metrics(
        per_jaw, pred_key="y_pred", prob_field="prob", tens_pred_key="y_tens_pred"
    )
    print("\nComplete vs missing:")
    for group, m in missing_vs_complete.items():
        print(f"{group:10s} n_arches={m['n_arches']:5d}  precision={m['precision']:.4f}  "
              f"recall={m['recall']:.4f}  f1={m['f1']:.4f}  acc={m['accuracy']:.4f}  auc={m['auc']:.4f}")

    complexity_build = load_module(
        "vit_complexity_build_eval_ghorbani", PROJECT_ROOT / "ViT" / "complexity" / "build_dataset.py"
    )
    complexity_by_image = complexity_build.load_complexity_labels(DATASET_DIR, "test")
    arch_by_complexity = tooth_module.compute_arch_metrics_by_complexity(
        per_jaw, complexity_by_image, pred_key="y_pred", prob_field="prob", tens_pred_key="y_tens_pred"
    )
    print("\nBy Arch Complexity class:")
    complexity_labels = {1: "Class I", 2: "Class II", 3: "Class III"}
    for c, m in sorted(arch_by_complexity.items()):
        print(f"{complexity_labels.get(c, c):10s} n_arches={m['n_arches']:5d}  "
              f"precision={m['precision']:.4f}  recall={m['recall']:.4f}  f1={m['f1']:.4f}  "
              f"acc={m['accuracy']:.4f}  auc={m['auc']:.4f}")

    with open(RESULTS_DIR / "ghorbani_arch_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "missing_teeth_metrics": missing,
            "missing_vs_complete": missing_vs_complete,
            "by_complexity": arch_by_complexity,
        }, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'ghorbani_arch_metrics.json'}")


if __name__ == "__main__":
    main()
