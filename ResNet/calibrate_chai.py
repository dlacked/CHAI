"""
Probability calibration (temperature scaling) for the proposed model's unified 12-way classifier,
applied BEFORE the Hungarian cost matrix (-log(prob)) is built - requested by the advisor to make
the classifier's raw softmax output statistically trustworthy in absolute value, not just
well-ranked per tooth, since Hungarian's duplicate-resolution logic compares confidences ACROSS
different teeth competing for the same class.

Temperature scaling: fit a single scalar T>0 on the VAL split (never the test split) by minimizing
NLL of softmax(logit/T) against the true class12 label, then apply that same T to the TEST split's
probabilities before re-running Hungarian assignment (ResNet/postprocess.py's resolve_arch_chai).

Reconstructing "logit/T" without needing the model's raw pre-softmax logits: since
ToothPositionClassifierNoTheta.forward() only returns already-softmaxed probabilities (see
train.py's forward(), which is shared via inheritance), this script instead works from log(prob).
This is mathematically exact for temperature scaling specifically, not an approximation: if
p = softmax(logit), then log(p) = logit - C for a per-sample constant C = logsumexp(logit)
(same constant subtracted from every entry of that sample's row). Because softmax is invariant to
adding a constant to every logit, softmax(logit/T) = softmax((log(p)+C)/T) = softmax(log(p)/T + C/T)
= softmax(log(p)/T) exactly - the unknown C/T term cancels in the softmax normalization. So
"temperature-scaling the logits" and "temperature-scaling log(p)" give identical results, letting
this script work entirely from the model's existing probability output.

Crops with the exact int()-truncated pad=10 box (same convention as evaluate_chai.py, NOT the
ResNet/cropped_tooth/ precrop cache - see that script's docstring for why the cache's
sub-pixel PIL float crop was rejected for evaluation use).

Usage:
    .venv/Scripts/python.exe ResNet/calibrate_chai.py
"""
import csv
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import minimize_scalar
from torchvision import transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TOOTH_DIR = Path(__file__).resolve().parent
_bootstrap_root = TOOTH_DIR.parent
if str(_bootstrap_root) not in sys.path:
    sys.path.append(str(_bootstrap_root))
from config import PathConfig
PROJECT_ROOT = PathConfig.ROOT
DATASET_DIR = PROJECT_ROOT.parent / "dataset"
MODEL_PATH = TOOTH_DIR / "model" / "best.pth"
RESULTS_DIR = TOOTH_DIR / "results"

if str(TOOTH_DIR) not in sys.path:
    sys.path.append(str(TOOTH_DIR))
from train_mirrored import ToothPositionClassifierNoTheta  # noqa: E402
from postprocess import resolve_arch_chai  # noqa: E402

JAW_TENS = {"upper": (1, 2), "lower": (3, 4)}

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_rows(split):
    csv_path = TOOTH_DIR / "csv" / f"features_{split}.csv"
    rows_by_arch = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["jaw"], row["image_name"])
            rows_by_arch.setdefault(key, []).append(row)
    return rows_by_arch


def get_full_probs(model, device, split):
    """Returns, per tooth: full (12,) raw softmax vector, true class12, jaw, image_name, fdi_number.
    One forward pass per arch (all its teeth batched together), same crop convention as
    evaluate_chai.py."""
    rows_by_arch = load_rows(split)
    image_dir = DATASET_DIR / split / "images"
    json_dir = DATASET_DIR / split / "labels_json"

    all_probs, all_true, all_jaw, all_image, all_fdi = [], [], [], [], []

    n_done = 0
    for (jaw, image_name), rows in rows_by_arch.items():
        stem = Path(image_name).stem
        image_full = cv2.imread(str(image_dir / jaw / image_name))
        if image_full is None:
            continue
        h, w = image_full.shape[:2]

        with open(json_dir / jaw / f"{stem}.json", "r", encoding="utf-8") as f:
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

        imgs, metas, class12_labels, fdi_list = [], [], [], []
        for row in rows:
            fdi = int(row["fdi_number"])
            if fdi not in box_by_fdi:
                continue
            x1, y1, x2, y2 = box_by_fdi[fdi]
            cropped = image_full[y1:y2, x1:x2]
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            imgs.append(val_transform(Image.fromarray(cropped_rgb)))
            metas.append([float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])])
            class12_labels.append(int(row["class12"]))
            fdi_list.append(fdi)

        if not imgs:
            continue

        img_batch = torch.stack(imgs).to(device)
        meta_batch = torch.tensor(metas, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = model(img_batch, meta_batch).cpu().numpy()

        for i in range(len(imgs)):
            all_probs.append(probs[i])
            all_true.append(class12_labels[i])
            all_jaw.append(jaw)
            all_image.append(image_name)
            all_fdi.append(fdi_list[i])

        n_done += 1
        if n_done % 1000 == 0:
            print(f"  [{split}] {n_done}/{len(rows_by_arch)} arches")

    return {
        "probs": np.array(all_probs), "true": np.array(all_true),
        "jaw": all_jaw, "image": all_image, "fdi": all_fdi,
    }


def softmax_from_logp(logp, T):
    scaled = logp / T
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(probs, true_labels):
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    n = len(true_labels)
    idx = np.arange(n)

    def nll(T):
        sm = softmax_from_logp(logp, T)
        return -np.mean(np.log(np.clip(sm[idx, true_labels], 1e-12, 1.0)))

    result = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(result.x), nll(1.0), nll(result.x)


def reliability_diagram(confidences, corrects, n_bins=10):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf, bin_acc, bin_count = [], [], []
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = mask.sum()
        if count == 0:
            bin_conf.append(np.nan)
            bin_acc.append(np.nan)
            bin_count.append(0)
            continue
        c = confidences[mask].mean()
        a = corrects[mask].mean()
        bin_conf.append(c)
        bin_acc.append(a)
        bin_count.append(int(count))
        ece += (count / n) * abs(a - c)
    return {"bin_edges": bin_edges.tolist(), "bin_confidence": bin_conf, "bin_accuracy": bin_acc,
            "bin_count": bin_count, "ece": float(ece)}


def per_arch_from_probs(rows_by_arch_probs, resolve_fn):
    """rows_by_arch_probs: {(jaw,image): (probs (n,12) array, true_fdi list)}.
    Returns per-arch exact-match booleans and pooled per-tooth accuracy."""
    arch_exact = {}
    tooth_correct, tooth_total = 0, 0
    for key, (probs, true_fdis) in rows_by_arch_probs.items():
        jaw = key[0]
        class12_preds = resolve_fn(probs)
        exact = True
        for i, fdi in enumerate(true_fdis):
            class12 = int(class12_preds[i])
            local_quadrant, digit0 = class12 // 6, class12 % 6
            pred_fdi = JAW_TENS[jaw][local_quadrant] * 10 + digit0 + 1
            correct = (pred_fdi == fdi)
            tooth_correct += int(correct)
            tooth_total += 1
            exact = exact and correct
        arch_exact[key] = exact
    accuracy_arch = sum(arch_exact.values()) / len(arch_exact)
    accuracy_tooth = tooth_correct / tooth_total
    return accuracy_arch, accuracy_tooth, arch_exact


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ToothPositionClassifierNoTheta(num_classes=12, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded {MODEL_PATH}")

    print("\n=== Extracting VAL split probabilities (for fitting T) ===")
    val_data = get_full_probs(model, device, "val")
    print(f"val: {len(val_data['true'])} teeth")

    T, nll_before, nll_after = fit_temperature(val_data["probs"], val_data["true"])
    print(f"\nFitted temperature T = {T:.4f}")
    print(f"Val NLL before calibration (T=1): {nll_before:.4f}")
    print(f"Val NLL after calibration  (T={T:.4f}): {nll_after:.4f}")

    print("\n=== Extracting TEST split probabilities ===")
    test_data = get_full_probs(model, device, "test")
    print(f"test: {len(test_data['true'])} teeth")

    raw_probs = test_data["probs"]
    calibrated_probs = softmax_from_logp(np.log(np.clip(raw_probs, 1e-12, 1.0)), T)

    # --- Reliability diagrams (top-1 argmax confidence vs correctness, per tooth, pre-Hungarian) ---
    raw_top1 = raw_probs.argmax(axis=1)
    raw_conf = raw_probs[np.arange(len(raw_top1)), raw_top1]
    raw_correct = (raw_top1 == test_data["true"]).astype(float)

    cal_top1 = calibrated_probs.argmax(axis=1)
    cal_conf = calibrated_probs[np.arange(len(cal_top1)), cal_top1]
    cal_correct = (cal_top1 == test_data["true"]).astype(float)
    # top-1 argmax is temperature-invariant (monotonic rescaling), so raw_top1 == cal_top1 always -
    # only the reported CONFIDENCE value (and therefore the reliability diagram / ECE) changes.

    rel_raw = reliability_diagram(raw_conf, raw_correct)
    rel_cal = reliability_diagram(cal_conf, cal_correct)
    print(f"\nRaw (T=1) ECE: {rel_raw['ece']:.4f}")
    print(f"Calibrated (T={T:.4f}) ECE: {rel_cal['ece']:.4f}")

    # --- Plot reliability diagram (before vs after) ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, rel, title in [(axes[0], rel_raw, f"Before Calibration (T=1.0), ECE={rel_raw['ece']:.4f}"),
                            (axes[1], rel_cal, f"After Calibration (T={T:.2f}), ECE={rel_cal['ece']:.4f}")]:
        bin_centers = [(rel["bin_edges"][i] + rel["bin_edges"][i + 1]) / 2 for i in range(10)]
        conf = [c if c == c else np.nan for c in rel["bin_confidence"]]
        acc = [a if a == a else np.nan for a in rel["bin_accuracy"]]
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        ax.bar(bin_centers, acc, width=0.08, alpha=0.7, edgecolor="black", label="Accuracy")
        ax.plot(bin_centers, conf, marker="o", color="red", label="Confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    out_fig = RESULTS_DIR / "calibration_reliability_diagram.png"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=150)
    print(f"Saved reliability diagram to {out_fig}")

    # --- Re-run Hungarian with raw vs calibrated probs, compare Exact-Match/F1 ---
    def build_arch_dict(probs_array):
        d = {}
        for i in range(len(test_data["true"])):
            key = (test_data["jaw"][i], test_data["image"][i])
            d.setdefault(key, ([], []))
            d[key][0].append(probs_array[i])
            d[key][1].append(test_data["fdi"][i])
        return {k: (np.array(v[0]), v[1]) for k, v in d.items()}

    raw_arch_dict = build_arch_dict(raw_probs)
    cal_arch_dict = build_arch_dict(calibrated_probs)

    acc_arch_raw, acc_tooth_raw, exact_raw = per_arch_from_probs(raw_arch_dict, resolve_arch_chai)
    acc_arch_cal, acc_tooth_cal, exact_cal = per_arch_from_probs(cal_arch_dict, resolve_arch_chai)

    print(f"\n=== Full test set (n={len(raw_arch_dict)} arches) ===")
    print(f"Raw:        arch Exact-Match={acc_arch_raw:.4f}  per-tooth Acc={acc_tooth_raw:.4f}")
    print(f"Calibrated: arch Exact-Match={acc_arch_cal:.4f}  per-tooth Acc={acc_tooth_cal:.4f}")

    # McNemar-style discordant count (raw vs calibrated), full test set
    keys = list(raw_arch_dict.keys())
    raw_only = sum(1 for k in keys if exact_raw[k] and not exact_cal[k])
    cal_only = sum(1 for k in keys if exact_cal[k] and not exact_raw[k])
    print(f"Discordant arches: raw-only-correct={raw_only}, calibrated-only-correct={cal_only}")

    results = {
        "temperature": T,
        "val_nll_before": nll_before, "val_nll_after": nll_after,
        "test_ece_before": rel_raw["ece"], "test_ece_after": rel_cal["ece"],
        "reliability_raw": rel_raw, "reliability_calibrated": rel_cal,
        "full_test": {
            "n_arches": len(raw_arch_dict),
            "raw": {"arch_exact_match": acc_arch_raw, "per_tooth_accuracy": acc_tooth_raw},
            "calibrated": {"arch_exact_match": acc_arch_cal, "per_tooth_accuracy": acc_tooth_cal},
            "discordant_raw_only": raw_only, "discordant_calibrated_only": cal_only,
        },
    }
    def to_native(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(RESULTS_DIR / "calibration_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=to_native)
    print(f"\nSaved to {RESULTS_DIR / 'calibration_results.json'}")

    # Save raw+calibrated arrays for further slicing (missing-tooth subset etc.) without rerunning inference
    with open(RESULTS_DIR / "calibration_test_probs.pkl", "wb") as f:
        pickle.dump({"test_data": test_data, "raw_probs": raw_probs,
                     "calibrated_probs": calibrated_probs, "temperature": T}, f)
    print(f"Saved raw+calibrated probability arrays to {RESULTS_DIR / 'calibration_test_probs.pkl'}")


if __name__ == "__main__":
    main()
