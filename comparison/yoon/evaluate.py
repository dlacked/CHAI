"""
Evaluation for the Yoon et al. reproduction: runs the trained Cascade R-CNN end-to-end (it detects
AND classifies each tooth's FDI number jointly, unlike Ghorbani's two-stage detect-then-classify
pipeline - see comparison/ghorbani/evaluate.py for that one), matches predictions to CHAI's GT via
IoU, and reuses functions/graph/tooth.py's arch-level metric functions so this model can be
dropped into the same {model_label: {group_key: metrics}} comparison dicts as CHAI and Ghorbani.

No duplicate-resolution post-process here (unlike Ghorbani's Fig.1 half-jaw reassignment step) -
Yoon et al.'s paper describes no such mechanism for their model, just "identified and localized
the bounding boxes for all teeth" and "recognized the number of teeth associated with each
detected tooth". Taking the Cascade R-CNN's own top-1 class per detection directly is therefore
the faithful reproduction, not an omission.

Usage:
    .venv/Scripts/python.exe comparison/yoon/evaluate.py [checkpoint_path]
    (checkpoint_path defaults to the best/latest checkpoint mmengine saved under runs/)
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

YOON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = YOON_DIR.parent.parent
DATASET_DIR = PROJECT_ROOT.parent / "dataset"
RESULTS_DIR = YOON_DIR / "results"

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".webp"]
CONF_THRESHOLD = 0.25
IOU_MATCH_THRESHOLD = 0.5

FDI_CLASSES = [q * 10 + d for q in (1, 2, 3, 4) for d in range(1, 7)]
FDI_TO_IDX = {n: i for i, n in enumerate(FDI_CLASSES)}


def find_checkpoint(explicit_path):
    if explicit_path:
        return explicit_path
    # Prefer the best-val-mAP checkpoint (see config.py's save_best='coco/bbox_mAP_50') over the
    # merely-latest epoch, matching the same "best, not last" convention used for Ghorbani/CHAI.
    best = sorted(glob.glob(str(YOON_DIR / "runs" / "best_coco_bbox_mAP_50_epoch_*.pth")))
    if best:
        return best[-1]
    latest = sorted(glob.glob(str(YOON_DIR / "runs" / "epoch_*.pth")),
                     key=lambda p: int(p.rsplit("_", 1)[-1].split(".")[0]))
    if latest:
        return latest[-1]
    raise SystemExit("No checkpoint found under runs/ - train.py hasn't completed an epoch yet.")


def poly_bbox(seg):
    if isinstance(seg[0], list) and len(seg[0]) == 2:
        poly = np.array(seg, dtype=float)
    else:
        poly = np.array(seg, dtype=float).reshape(-1, 2)
    return poly.min(axis=0), poly.max(axis=0)


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


def iou_matrix(boxes_a, boxes_b):
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


def evaluate_split(model, split="test"):
    from mmdet.apis import inference_detector

    per_jaw = {}
    totals = {"tp": 0, "fp": 0, "fn": 0, "spurious": 0}

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

            result = inference_detector(model, str(image_dir / image_name))
            pred = result.pred_instances
            keep = (pred.scores >= CONF_THRESHOLD).cpu().numpy()
            det_boxes = pred.bboxes.cpu().numpy()[keep].tolist()
            det_scores = pred.scores.cpu().numpy()[keep]
            det_labels = pred.labels.cpu().numpy()[keep]

            matched, unmatched_det, unmatched_gt = match_detections_to_gt(det_boxes, gt_boxes)
            matched_gt_idx = {gt_i: det_i for det_i, gt_i in matched}

            for gt_i, fdi_number in enumerate(gt_fdi):
                y_true.append(fdi_number % 10 - 1)
                y_fdi_number.append(fdi_number)
                y_image_name.append(image_name)

                if gt_i in matched_gt_idx:
                    det_i = matched_gt_idx[gt_i]
                    pred_class = FDI_CLASSES[int(det_labels[det_i])]
                    pred_tens, pred_digit = pred_class // 10, pred_class % 10 - 1
                    y_pred.append(pred_digit)
                    y_tens_pred.append(pred_tens)
                    conf_vec = [0.0] * 6
                    conf_vec[pred_digit] = float(det_scores[det_i])
                    prob_field.append(conf_vec)
                    totals["tp" if pred_class == fdi_number else "fp"] += 1
                else:
                    # Missed GT tooth: guaranteed-wrong sentinel, same convention as
                    # comparison/ghorbani/evaluate.py (tens=0 never matches a real 1-4 tens digit).
                    y_pred.append(0)
                    y_tens_pred.append(0)
                    prob_field.append([0.0] * 6)
                    totals["fn"] += 1

            totals["spurious"] += len(unmatched_det)

            if (i + 1) % 50 == 0 or (i + 1) == len(json_paths):
                print(f"  [{jaw}] {i + 1}/{len(json_paths)} images")

        per_jaw[jaw] = {
            "y_true": y_true, "y_pred": y_pred, "y_tens_pred": y_tens_pred,
            "prob": prob_field, "y_fdi_number": y_fdi_number, "y_image_name": y_image_name,
            "n_teeth": len(y_true),
        }
        print(f"{jaw}: {len(y_true)} GT teeth scored.")

    return per_jaw, totals


def load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    from mmdet.apis import init_detector

    checkpoint = find_checkpoint(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Checkpoint: {checkpoint}")
    model = init_detector(str(YOON_DIR / "config.py"), checkpoint, device="cuda:0")

    per_jaw, totals = evaluate_split(model, "test")

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    print("\n" + "=" * 70)
    print("Detection-level metrics (test set, IoU>=0.5 match)")
    print("=" * 70)
    print(f"TP={tp}  FP={fp}  FN={fn}  spurious(no-GT-match)={totals['spurious']}")
    print(f"Sensitivity={sensitivity:.4f}  Precision={precision:.4f}  F1={f1:.4f}")

    tooth_module = load_module("chai_tooth_graph_yoon", PROJECT_ROOT / "functions" / "graph" / "tooth.py")

    print("\n" + "=" * 70)
    print("CHAI-style arch-level metrics (test set) - Yoon (Cascade R-CNN)")
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
        "vit_complexity_build_eval_yoon", PROJECT_ROOT / "Transformer" / "complexity" / "build_dataset.py"
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "yoon_arch_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": str(checkpoint),
            "detection_level": {"tp": tp, "fp": fp, "fn": fn, "spurious": totals["spurious"],
                                 "sensitivity": sensitivity, "precision": precision, "f1": f1},
            "missing_teeth_metrics": missing,
            "missing_vs_complete": missing_vs_complete,
            "by_complexity": arch_by_complexity,
        }, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'yoon_arch_metrics.json'}")


if __name__ == "__main__":
    main()
