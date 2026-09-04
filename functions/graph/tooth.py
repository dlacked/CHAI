"""
Evaluates the tooth-number pipeline's final prediction (is the predicted FDI last digit the
true one?) on the held-out test split (dataset/test/, never touched by training or validation).

Reuses the project's existing pipelines instead of reimplementing them:
functions/features/main.py (GT feature CSV extraction), backups/superseded_20260831/ViT_arch/
build_dataset.py (ResNet crop/feature/probability extraction per arch - retired, see below),
and ResNet/tooth/postprocess.py (the per-quadrant Hungarian duplicate-resolution post-process
that replaced this old arch-refinement Transformer - see that module's docstring for why).

NOTE (2026-08-31): this whole module is currently BROKEN if run directly - CSV_DIR/RESNET_MODEL_DIR/
ARCH_MODEL_DIR below point at paths that were moved to backups/superseded_20260831/ during the
Baseline/Comp1-3 cleanup. Its individual functions (compute_missing_vs_complete_metrics etc.) are
still imported and reused directly by ad-hoc eval scripts against the new Baseline/Comp1/Comp2/Comp3
models - don't "fix" the module-level paths without checking those call sites first.

Leads with macro F1 rather than accuracy: with the last-digit classes this imbalanced (incisors
vastly outnumber missing/rare positions), accuracy alone can look flat even as rare-class
performance moves - F1 is the metric worth watching first.

Usage:
    .venv/Scripts/python.exe functions/graph/tooth.py
"""
import csv as csv_module
import importlib.util
import json
import os
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)

# All 32 FDI tooth numbers (8 per quadrant), in reading order for the per-number bar charts
ALL_FDI_NUMBERS = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28,
                   31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]

GRAPH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GRAPH_DIR.parent.parent          # .../CHAI/CHAI
DATASET_DIR = PROJECT_ROOT.parent / "dataset"   # .../CHAI/dataset
RESULTS_DIR = GRAPH_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Baseline (paper label, "CHAI (Ours)") as of 2026-08-31 - was variant A's ResNet/tooth/{csv,model}
# (now backups/superseded_20260831/), fixed to point here so evaluate_tooth_number() actually
# evaluates the model the paper reports, not a retired one. csv_baseline is ONE combined
# (both-jaws, `jaw` column) file per split, not per-jaw files - see build_baseline_arch_sequences.
CSV_DIR = PROJECT_ROOT / "ResNet" / "tooth" / "csv_baseline"
RESNET_MODEL_DIR = PROJECT_ROOT / "ResNet" / "tooth" / "model_baseline"
ARCH_MODEL_DIR = PROJECT_ROOT / "backups" / "superseded_20260831" / "ViT_arch" / "model"  # retired

# Metrics in recommended-first order - every plot/print below shows whichever of these are
# present, F1 leading, rather than defaulting to accuracy as the headline number.
METRIC_DISPLAY = [("f1", "F1 (Macro)"), ("accuracy", "Accuracy"),
                   ("precision", "Precision"), ("recall", "Recall")]

# dataviz reference palette - fixed categorical order (blue = slot 1, orange = slot 2)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
PURPLE = "#8858c8"
TEAL = "#1f9d8a"
PINK = "#d6538a"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# The five prediction sources evaluate_tooth_number() computes, in comparison-chart order:
# (key, display label, per-tooth hard-label field, per-tooth probability field, per-tooth tens-
# prediction field). AUC for the three post-process methods is scored against `prob_resnet` (see
# compute_method_comparison's docstring for why that's correct, not an oversight).
# resnet_hungarian uniquely gets its own tens field (y_tens_pred_hungarian): it resolves digits
# arch-wide before touching quadrant/tens at all (resolve_arch_duplicates), then reads the tens
# boundary off where a digit repeats (correct_mirrors_by_digit_occurrence) - see
# evaluate_tooth_number - so it has a materially better tens guess than the shared geometry-only
# one (y_tens_pred) every other method here is stuck with.
METHODS = [
    ("resnet_only", "ResNet-only", "y_resnet", "prob_resnet", "y_tens_pred"),
    ("resnet_vit", "ResNet+ViT", "y_vit", "prob_vit", "y_tens_pred"),
    ("resnet_hungarian", "ResNet+Hungarian", "y_postproc", "prob_resnet", "y_tens_pred_hungarian"),
    ("resnet_monodp", "ResNet+Monotonic-DP", "y_monodp", "prob_resnet", "y_tens_pred"),
    ("resnet_greedy", "ResNet+Greedy", "y_greedy", "prob_resnet", "y_tens_pred"),
]
METHOD_COLORS = {
    "resnet_only": BLUE, "resnet_vit": ORANGE, "resnet_hungarian": PURPLE,
    "resnet_monodp": TEAL, "resnet_greedy": PINK,
}

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


def plot_metric_bars(metrics, title, out_path):
    present = [(key, label) for key, label in METRIC_DISPLAY if key in metrics]
    labels = [label for _, label in present]
    values = [metrics[key] for key, _ in present]

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
    """Stacked small multiples (one per metric present), one bar per FDI tooth number that
    appeared in the test set, in fixed reading order (quadrant 1 -> 2 -> 3 -> 4) so the same
    position lands in the same spot across charts."""
    present_metrics = [(key, label) for key, label in METRIC_DISPLAY if key in metrics_by_number]
    present_numbers = [n for n in ALL_FDI_NUMBERS
                        if any(n in metrics_by_number[key] for key, _ in present_metrics)]
    labels = [str(n) for n in present_numbers]

    fig, axes = plt.subplots(len(present_metrics), 1, figsize=(14, 4 * len(present_metrics)), facecolor=SURFACE)
    if len(present_metrics) == 1:
        axes = [axes]
    for ax, (key, mtitle) in zip(axes, present_metrics):
        values = [metrics_by_number[key].get(n, 0.0) for n in present_numbers]
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


# Sentinel "digit" for a tooth whose predicted FDI *tens* digit was wrong (see
# compute_tens_predictions / gate_by_tens) - distinct from any real last-digit class (0-5), so a
# quadrant-assignment miss shows up as its own bucket instead of being silently absorbed into
# whichever last digit the model happened to guess (or, the opposite failure mode, silently
# scored as "correct" just because the last digit matched while the tooth was attributed to the
# wrong tooth position entirely).
WRONG_QUADRANT = 6
DIGIT_DISPLAY_LABELS = ["1", "2", "3", "4", "5", "6", "Wrong\nQuadrant"]


def gate_by_tens(y_pred, y_tens_pred, y_fdi_number):
    """Replaces a predicted digit with the WRONG_QUADRANT sentinel wherever this tooth's
    predicted FDI tens digit (y_tens_pred) didn't match the true one (y_fdi_number // 10). Every
    accuracy/F1/confusion-matrix view in this module scores against the gated prediction, not
    the raw digit, because a wrong quadrant makes the FDI number wrong even if the model's last
    digit was correct - previously that case was invisibly scored as a hit (see
    compute_tens_predictions's docstring for the actual bug this exposed).

    Takes the tens prediction as an explicit argument rather than a precomputed correctness
    array because different methods can have different tens predictions - resnet_hungarian's
    (see evaluate_tooth_number) is resolved from its own already-deduplicated digits via
    correct_mirrors_by_digit_occurrence, not the shared geometry-only guess every other method
    uses, so gating it against the plain geometric guess would understate how often it's
    actually right."""
    return [p if tp == (n // 10) else WRONG_QUADRANT
            for p, tp, n in zip(y_pred, y_tens_pred, y_fdi_number)]


def compute_tens_predictions(dataset_dir, jaw, split):
    """PCA-based FDI tens-digit (quadrant) guess - PCA over every detected tooth centroid in the
    arch, then each tooth's own rotated-x sign decides its quadrant. SUPERSEDED as of 2026-08-31:
    production now uses the width/2-based guess instead (see tens_norot.py's
    compute_tens_predictions_norot, and js/geometry.js computeToothMetaPooled) - kept here only
    as the ablation comparison point (evaluate_tooth_number no longer calls this).

    The last digit is the only thing the model predicts; the tens digit has always come from a
    geometric step like this one, both in production and, silently, in every eval chart in this
    module before compute_tens_predictions existed - which used the *ground-truth* tens digit
    instead, i.e. assumed the geometric step is always correct. It isn't: a missing tooth skews
    the PCA-estimated midline, and the tooth most exposed to that skew is exactly the one closest
    to it (a digit-"1" tooth) - see project notes for a live example (a real 41 and 31 both
    reading as "31") - exactly the failure mode width/2 (immune to which teeth are present, by
    construction) doesn't have.

    Returns {(image_name, fdi_number): predicted_tens}.
    """
    theta_module = load_module("features_theta", PROJECT_ROOT / "functions" / "features" / "theta.py")
    calculate_pca_rotation = theta_module.calculate_pca_rotation

    image_exts = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".webp"]
    json_dir = Path(dataset_dir) / split / "labels_json" / jaw
    image_dir = Path(dataset_dir) / split / "images" / jaw

    result = {}
    for json_path in sorted(json_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        segs = []
        for t in data.get("tooth", []):
            num = t.get("teeth_num")
            seg = t.get("segmentation", [])
            if num is None or not seg:
                continue
            poly = seg if (isinstance(seg[0], list) and len(seg[0]) == 2) else \
                [[seg[i], seg[i + 1]] for i in range(0, len(seg), 2)]
            cx = sum(p[0] for p in poly) / len(poly)
            cy = sum(p[1] for p in poly) / len(poly)
            segs.append((num, cx, cy))
        if len(segs) < 2:
            continue

        centroids = [(cx, cy) for _, cx, cy in segs]
        mean_pt, angle = calculate_pca_rotation(centroids)
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)

        image_name = None
        for ext in image_exts:
            if (image_dir / f"{json_path.stem}{ext}").exists():
                image_name = f"{json_path.stem}{ext}"
                break
        if image_name is None:
            continue

        for num, cx, cy in segs:
            rx = (cx - mean_pt[0]) * cos_a - (cy - mean_pt[1]) * sin_a
            mirrored = rx >= 0
            if jaw == "upper":
                tens = 2 if mirrored else 1
            else:
                tens = 3 if mirrored else 4
            result[(image_name, num)] = tens

    return result


# ---------------------------------------------------------------------------
# Tooth number (FDI last digit) performance - ResNet-only vs ResNet+Hungarian
# ---------------------------------------------------------------------------
def load_fdi_numbers_per_arch(csv_path, max_teeth_per_arch=12):
    """Reproduces ToothDataset's row filter and build_split's per-image grouping/truncation
    (backups/superseded_20260831/ViT_arch/build_dataset.py) using the FDI number column that build_split's own return
    value drops, so the result lines up 1:1 with arch_build.build_split's `sequences` list -
    lets per-tooth predictions be attributed back to a specific FDI number.

    NOTE (2026-08-31): evaluate_tooth_number() no longer calls this - it builds fdi_number lists
    itself, in the same pass as the ResNet forward pass, inside build_baseline_arch_sequences
    (needed since csv_baseline is one combined-jaws CSV, not per-jaw files, so grouping needs a
    jaw filter first anyway - doing both in one pass avoids filtering twice). Left here, unused,
    in case anything still wants a standalone FDI-numbers-per-arch reader for a per-jaw CSV."""
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


def build_baseline_arch_sequences(jaw, split, dataset_dir, csv_dir, model_dir, device, max_teeth_per_arch=12):
    """Baseline counterpart to the old (variant-A) arch_build.build_split
    (backups/superseded_20260831/ViT_arch/build_dataset.py): runs the Baseline ResNet model
    (ToothPositionClassifierNoTheta, train_baseline.py - 4-dim meta, no theta, pooled both-jaws)
    over one jaw's arches, grouped by image_name, and returns per-tooth softmax probabilities in
    the same arch-sequence shape evaluate_tooth_number() expects.

    csv_baseline is ONE combined (`jaw` column) CSV per split - filtered to this jaw here (same
    trick as Transformer/complexity/build_dataset.py and ResNet/tooth/train_comp2.py use), and
    since main_baseline.py writes one jaw's rows fully before starting the other, filtering still
    leaves each image's rows contiguous and in Held-Karp arch order.

    No `geom`/`img_vec` fields in the returned sequences (unlike the old build_split) - those fed
    the old arch-refinement Transformer's own two extra input branches, which has no
    Baseline-compatible equivalent, so evaluate_tooth_number()'s y_vit column falls back to
    y_resnet for every arch (ARCH_MODEL_DIR / f"{jaw}_best.pth" simply won't exist under
    Baseline's naming, so that fallback triggers automatically - see that function's arch_model
    handling).

    Returns (sequences, fdi_number_lists):
      sequences: [{"image_name": str, "target": LongTensor[n], "prob_vec": FloatTensor[n, 6]}, ...]
      fdi_number_lists: parallel list of this arch's true FDI numbers, same order as target.
    """
    train_baseline_module = load_module(
        "train_baseline_eval", PROJECT_ROOT / "ResNet" / "tooth" / "train_baseline.py"
    )
    ToothPositionClassifierNoTheta = train_baseline_module.ToothPositionClassifierNoTheta

    csv_path = Path(csv_dir) / f"features_{split}.csv"
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv_module.DictReader(f):
            if row.get("jaw") != jaw:
                continue
            fdi_digit = row.get("fdi_last_digit")
            if fdi_digit not in (None, "") and 1 <= int(fdi_digit) <= 6:
                rows.append(row)

    groups = OrderedDict()
    for row in rows:
        groups.setdefault(row["image_name"], []).append(row)

    model = ToothPositionClassifierNoTheta(num_classes=6, pretrained=False)
    model.load_state_dict(torch.load(Path(model_dir) / "best.pth", map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    sequences = []
    fdi_number_lists = []
    with torch.no_grad():
        for image_name, group_rows in groups.items():
            if len(group_rows) > max_teeth_per_arch:
                group_rows = group_rows[:max_teeth_per_arch]

            img_path = Path(dataset_dir) / split / "images" / jaw / image_name
            image = cv2.imread(str(img_path))
            if image is None:
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            h, w = image.shape[:2]

            image_name_no_ext = os.path.splitext(image_name)[0]
            json_path = Path(dataset_dir) / split / "labels_json" / jaw / f"{image_name_no_ext}.json"
            polys_by_fdi = {}
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for t in data.get("tooth", []):
                    num = t.get("teeth_num")
                    seg = t.get("segmentation", [])
                    if num is None or not seg:
                        continue
                    poly = np.array(seg) if (isinstance(seg[0], list) and len(seg[0]) == 2) else np.array(seg).reshape(-1, 2)
                    polys_by_fdi[num] = poly

            imgs, metas, targets, fdi_numbers = [], [], [], []
            for row in group_rows:
                fdi_number = int(row["fdi_number"])
                fdi_digit = int(row["fdi_last_digit"])

                poly = polys_by_fdi.get(fdi_number)
                if poly is not None:
                    x_min, y_min = poly.min(axis=0)
                    x_max, y_max = poly.max(axis=0)
                    pad = 10
                    x1 = int(max(0, x_min - pad)); y1 = int(max(0, y_min - pad))
                    x2 = int(min(w, x_max + pad)); y2 = int(min(h, y_max + pad))
                else:
                    x1, y1, x2, y2 = 0, 0, w, h

                cropped = image[y1:y2, x1:x2]
                if cropped.size == 0:
                    cropped = np.zeros((224, 224, 3), dtype=np.uint8)
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                imgs.append(transform(Image.fromarray(cropped_rgb)))
                metas.append([float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])])
                targets.append(fdi_digit - 1)
                fdi_numbers.append(fdi_number)

            img_tensor = torch.stack(imgs).to(device)
            meta_tensor = torch.tensor(metas, dtype=torch.float32).to(device)
            prob_vec = model(img_tensor, meta_tensor).cpu()  # forward() already applies softmax

            sequences.append({
                "image_name": image_name,
                "target": torch.tensor(targets, dtype=torch.long),
                "prob_vec": prob_vec,
            })
            fdi_number_lists.append(fdi_numbers)

    return sequences, fdi_number_lists


def evaluate_tooth_number():
    """Runs every method being compared over the test set in one pass (one ResNet forward pass
    per jaw is the expensive part - everything downstream of `prob_vec` is cheap, so it's not
    worth splitting into separate eval functions per method):

      - y_resnet    : ResNet-only, independent per-tooth argmax (today's fallback whenever a
                      jaw has no arch weights).
      - y_vit       : ResNet + the old arch-refinement Transformer this pipeline used to run in
                      production (see backups/superseded_20260831/ViT_arch/model.py) - kept only
                      for this comparison, not used anywhere else anymore (see
                      ResNet/tooth/postprocess.py's docstring for why it was retired in favor of
                      the Hungarian post-process below).
      - y_postproc  : ResNet + arch-wide Hungarian digit resolution, tens read off afterward by
                      digit-pairing - today's actual production pipeline (server.py
                      /tooth_predict). Digits first, tens second (resolve_arch_duplicates then
                      correct_mirrors_by_digit_occurrence) rather than the other three post-
                      processes' quadrant-first order, specifically because grouping by a
                      possibly-wrong tens guess *before* resolving digits let a bad quadrant
                      call corrupt an already-correct neighboring tooth's digit too - see project
                      notes for a caught example (a correct "1" reassigned to "5" this way).
      - y_monodp    : ResNet + the stricter monotonic-order DP - a comparison point, not used in
                      production (see resolve_quadrant_monotonic's docstring for why it lost).
      - y_greedy    : ResNet + greedy confidence-first duplicate resolution - a second,
                      independent post-process family (local heuristic vs. Hungarian's globally
                      optimal assignment), also not used in production.
    """
    postprocess = load_module("resnet_tooth_postprocess", PROJECT_ROOT / "ResNet" / "tooth" / "postprocess.py")
    tens_norot_module = load_module("tens_norot_eval", GRAPH_DIR / "tens_norot.py")

    per_jaw = {}
    for jaw in ("lower", "upper"):
        csv_path = CSV_DIR / "features_test.csv"
        if not csv_path.exists():
            print(f"Error: {csv_path} not found. Run functions/features/main_baseline.py first.")
            continue

        print(f"\nBuilding {jaw}/test arch sequences (Baseline ResNet forward pass)...")
        sequences, fdi_number_lists = build_baseline_arch_sequences(
            jaw, "test", str(DATASET_DIR), str(CSV_DIR), str(RESNET_MODEL_DIR), device
        )
        # Production's actual (geometry-only, occasionally wrong) tens-digit guess - used below
        # both to gate correctness (gate_by_tens) and, faithfully to production, as the grouping
        # key the quadrant-dedup post-processes (Hungarian etc.) actually see - not the GT tens,
        # which would hand them cleaner groups than production ever gets. As of 2026-08-31,
        # production's geometric guess is width/2-based, not PCA-based (see tens_norot.py and
        # js/geometry.js computeToothMetaPooled) - compute_tens_predictions (PCA) is now the old,
        # superseded comparison point, kept for the ablation table.
        tens_pred_map = tens_norot_module.compute_tens_predictions_norot(str(DATASET_DIR), jaw, "test")

        # arch_model is ALWAYS None now (was: loaded from ARCH_MODEL_DIR if weights existed there).
        # The old arch-refinement Transformer (backups/superseded_20260831/ViT_arch/model.py)
        # takes a DIFFERENT geometry format as input (5-dim PCA-rotated [x1,y1,x2,y2,theta] +
        # its own img_vec branch) than build_baseline_arch_sequences produces (4-dim raw
        # [x1,y1,x2,y2], no img_vec) - feeding it Baseline's geometry would be silently wrong, not
        # just "missing," so this comparison is disabled entirely rather than attempted with
        # mismatched inputs. y_vit falls back to y_resnet for every arch (matching the existing
        # eval_baseline_methods.py convention noted in project notes) until/unless a
        # Baseline-compatible arch-refinement model exists to compare against.
        arch_model = None
        print(f"Old arch-refinement Transformer disabled for {jaw} (incompatible input format, see comment above) - "
              f"ResNet+ViT column falls back to ResNet-only.")

        y_true, y_fdi_number, y_image_name = [], [], []
        y_resnet, y_vit, y_postproc, y_monodp, y_greedy = [], [], [], [], []
        y_tens_pred, y_tens_pred_hungarian = [], []
        prob_resnet, prob_vit = [], []
        with torch.no_grad():
            for seq, fdi_numbers in zip(sequences, fdi_number_lists):
                target = seq["target"]
                prob_vec = seq["prob_vec"]
                probs_np = prob_vec.numpy()

                y_true.extend(target.tolist())
                y_fdi_number.extend(fdi_numbers)
                y_image_name.extend([seq["image_name"]] * len(target))
                y_resnet.extend(torch.argmax(prob_vec, dim=1).tolist())
                prob_resnet.extend(probs_np.tolist())

                # Falls back to the true tens digit (i.e. assumes correct) only for the rare
                # tooth compute_tens_predictions has no geometry for (e.g. a single-tooth arch,
                # where PCA over one point is degenerate) - never silently drops a tooth.
                tens_pred = [tens_pred_map.get((seq["image_name"], n), n // 10) for n in fdi_numbers]
                y_tens_pred.extend(tens_pred)

                if arch_model is not None:
                    k = prob_vec.size(0)
                    key_padding_mask = torch.zeros(1, k, dtype=torch.bool, device=device)
                    refined_logits = arch_model(
                        seq["geom"].unsqueeze(0).to(device),
                        seq["img_vec"].unsqueeze(0).to(device),
                        prob_vec.unsqueeze(0).to(device),
                        key_padding_mask,
                    )
                    refined_probs = torch.softmax(refined_logits[0], dim=1).cpu()
                    y_vit.extend(torch.argmax(refined_probs, dim=1).tolist())
                    prob_vit.extend(refined_probs.tolist())
                else:
                    y_vit.extend(torch.argmax(prob_vec, dim=1).tolist())
                    prob_vit.extend(probs_np.tolist())

                # Production Hungarian (y_postproc): resolve digits arch-wide FIRST, with no
                # quadrant/tens information at all (resolve_arch_duplicates), then read the tens
                # boundary off wherever a digit repeats (correct_mirrors_by_digit_occurrence),
                # falling back to the geometry-only guess (tens_pred) only for a digit that
                # appears just once. See evaluate_tooth_number's docstring for why this order -
                # doing it the other way (quadrant-group first, like the two comparison post-
                # processes below still do) lets a bad quadrant call corrupt an already-correct
                # neighboring tooth's digit too.
                resolved_digits = postprocess.resolve_arch_duplicates(probs_np)
                initial_mirrors = [tp in (2, 3) for tp in tens_pred]
                corrected_mirrors = postprocess.correct_mirrors_by_digit_occurrence(
                    resolved_digits.tolist(), initial_mirrors
                )
                if jaw == "upper":
                    tens_hungarian = [2 if m else 1 for m in corrected_mirrors]
                else:
                    tens_hungarian = [3 if m else 4 for m in corrected_mirrors]
                y_postproc.extend(resolved_digits.tolist())
                y_tens_pred_hungarian.extend(tens_hungarian)

                # Comparison-only post-processes (not production): still quadrant-group first,
                # by the same geometry-only tens_pred production's OLD approach used - kept as-is
                # so the difference above is actually visible in the method comparison, not
                # quietly fixed everywhere at once.
                y_monodp.extend(postprocess.resolve_quadrant_monotonic(probs_np, tens_pred).tolist())
                y_greedy.extend(postprocess.resolve_quadrant_greedy(probs_np, tens_pred).tolist())

        per_jaw[jaw] = {
            "y_true": y_true,
            "y_resnet": y_resnet, "y_vit": y_vit, "y_postproc": y_postproc,
            "y_monodp": y_monodp, "y_greedy": y_greedy,
            "prob_resnet": prob_resnet, "prob_vit": prob_vit,
            "y_tens_pred": y_tens_pred, "y_tens_pred_hungarian": y_tens_pred_hungarian,
            "y_fdi_number": y_fdi_number, "y_image_name": y_image_name,
            "n_arches": len(sequences), "n_teeth": len(y_true),
        }
        print(f"{jaw}: {len(sequences)} arches, {len(y_true)} teeth.")

    return per_jaw


def plot_tooth_number_confusion(per_jaw, pred_key, label, out_path_template, tens_pred_key="y_tens_pred"):
    """Plots one confusion matrix per jaw for a single prediction source (pred_key is
    'y_resnet' for the ResNet-only baseline or 'y_postproc' for the pipeline's final,
    Hungarian-corrected output - both are already collected per-tooth in
    evaluate_tooth_number(), just not both plotted before). Called twice from main() so the two
    are directly comparable side by side.

    Gated by gate_by_tens before plotting - a tooth whose predicted FDI tens digit was wrong gets
    its digit prediction replaced with the WRONG_QUADRANT sentinel rather than scored against
    whatever digit the model happened to guess, so a wrong-quadrant miss shows up as its own
    column instead of silently landing on the diagonal (if the digit also happened to match) or
    polluting genuine digit-vs-digit confusion (if it didn't). tens_pred_key picks which tens
    prediction to gate against - resnet_hungarian gets its own, better one (see METHODS)."""
    for jaw, data in per_jaw.items():
        gated_pred = gate_by_tens(data[pred_key], data[tens_pred_key], data["y_fdi_number"])
        cm = confusion_matrix(data["y_true"], gated_pred, labels=list(range(7)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DIGIT_DISPLAY_LABELS)
        fig, ax = plt.subplots(figsize=(7.8, 6), facecolor=SURFACE)
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True, text_kw={"fontsize": 16})
        ax.set_title(f"Tooth Number Confusion Matrix - {jaw.capitalize()} Jaw ({label})", color=INK, fontsize=13)
        ax.set_xlabel(ax.get_xlabel(), fontsize=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)
        ax.tick_params(labelsize=11)
        plt.tight_layout()
        out_path = Path(str(out_path_template).format(jaw=jaw))
        plt.savefig(out_path, dpi=150, facecolor=SURFACE)
        plt.close()


def compute_tooth_number_metrics(per_jaw, pred_key, tens_pred_key="y_tens_pred"):
    """F1 (macro, primary)/accuracy/precision/recall over the 6 last-digit classes for one
    prediction source (pred_key: 'y_resnet' for the ResNet-only baseline, 'y_postproc' for the
    pipeline's final Hungarian-corrected output) against the GT last digit, pooled across both
    jaws. Gated by gate_by_tens - a tooth only counts as correct if BOTH its predicted last digit
    and its predicted FDI tens digit (tens_pred_key - production's geometry-only guess for most
    methods, resnet_hungarian's own better one for that method, see METHODS) match;
    `labels=range(6)` on the macro metrics keeps the WRONG_QUADRANT sentinel out of the per-class
    average itself while still letting it correctly zero out that instance's contribution to its
    true class's recall."""
    all_true, all_pred = [], []
    for jaw in ("lower", "upper"):
        d = per_jaw[jaw]
        all_true += d["y_true"]
        all_pred += gate_by_tens(d[pred_key], d[tens_pred_key], d["y_fdi_number"])

    digit_labels = list(range(6))
    return {
        "f1": f1_score(all_true, all_pred, labels=digit_labels, average="macro", zero_division=0),
        "accuracy": accuracy_score(all_true, all_pred),
        "precision": precision_score(all_true, all_pred, labels=digit_labels, average="macro", zero_division=0),
        "recall": recall_score(all_true, all_pred, labels=digit_labels, average="macro", zero_division=0),
    }


def compute_method_comparison(per_jaw):
    """Accuracy / F1 (macro) / AUC (macro, one-vs-rest) for every method in METHODS, per jaw and
    pooled.

    AUC needs class probability *scores*, not just the hard label a method settles on.
    ResNet-only and ResNet+ViT each have their own genuine softmax distribution over the 6
    digits. The three post-processes (Hungarian / Monotonic-DP / Greedy), by contrast, only ever
    pick which class an already-fixed ResNet softmax argmaxes to under some constraint - they
    don't produce a new probability distribution, so there is no other honest score to rank them
    by than the same ResNet probabilities resnet-only uses. That's why those three end up with
    identical AUC to ResNet-only below: it's not a bug, it reflects that none of them change the
    model's confidence estimates, only its final hard decision - accuracy and F1 (which DO differ
    across them, since they act on the hard label) are the metrics that actually show each
    post-process's effect.

    Accuracy/F1 are gated by gate_by_tens (see compute_tooth_number_metrics) - a method only gets
    credit when the predicted FDI tens digit was also right, using each method's own tens
    prediction field (METHODS - resnet_hungarian's is materially better than the shared
    geometry-only one every other method here uses). AUC is left ungated: it scores the ResNet
    softmax's ranking quality for the 6 digit classes the model was actually trained on, which
    the (geometry-only, model-independent) tens digit has no bearing on - gating it would just
    make every method's AUC dip by the same tens-error rate without reflecting anything about the
    classifier itself.
    """
    results = {}
    for series in ("lower", "upper", "pooled"):
        if series == "pooled":
            y_true = per_jaw["lower"]["y_true"] + per_jaw["upper"]["y_true"]
            y_fdi_number = per_jaw["lower"]["y_fdi_number"] + per_jaw["upper"]["y_fdi_number"]
        else:
            y_true = per_jaw[series]["y_true"]
            y_fdi_number = per_jaw[series]["y_fdi_number"]

        series_result = {}
        for key, label, pred_field, prob_field, tens_pred_field in METHODS:
            if series == "pooled":
                y_pred = per_jaw["lower"][pred_field] + per_jaw["upper"][pred_field]
                y_prob = per_jaw["lower"][prob_field] + per_jaw["upper"][prob_field]
                y_tens_pred = per_jaw["lower"][tens_pred_field] + per_jaw["upper"][tens_pred_field]
            else:
                y_pred = per_jaw[series][pred_field]
                y_prob = per_jaw[series][prob_field]
                y_tens_pred = per_jaw[series][tens_pred_field]
            y_pred_gated = gate_by_tens(y_pred, y_tens_pred, y_fdi_number)

            try:
                auc = roc_auc_score(y_true, np.array(y_prob), multi_class="ovr",
                                     average="macro", labels=list(range(6)))
            except ValueError:
                auc = float("nan")  # a class is entirely absent from this series - can't rank it

            series_result[key] = {
                "label": label,
                "accuracy": accuracy_score(y_true, y_pred_gated),
                "precision": precision_score(y_true, y_pred_gated, labels=list(range(6)), average="macro", zero_division=0),
                "recall": recall_score(y_true, y_pred_gated, labels=list(range(6)), average="macro", zero_division=0),
                "f1": f1_score(y_true, y_pred_gated, labels=list(range(6)), average="macro", zero_division=0),
                "auc": auc,
            }
        results[series] = series_result
    return results


def compute_arch_level_metrics(per_jaw):
    """Accuracy / F1 / AUC for every method in METHODS, per jaw and pooled - same as
    compute_method_comparison, but with one ARCH IMAGE as the scoring unit instead of one tooth.
    compute_method_comparison already answers "how many individual teeth did this method get
    right"; this answers the coarser, clinically-relevant question "how many whole arch images
    came back with every tooth right", which per-tooth accuracy can look excellent on while still
    missing every arch that has exactly one bad tooth.

    Accuracy (Exact Match Ratio): fraction of arches where every tooth's gate_by_tens-gated
    prediction matches the true digit - i.e. the entire FDI numbering for that arch is correct,
    not just most of it.

    F1 (sample-averaged / example-based, over FDI-number sets): each arch is treated as one
    multi-label instance whose "labels" are the FDI numbers present in it. Precision/recall are
    computed per arch from the overlap between its true FDI-number set and its predicted one
    (predicted numbers reconstructed from this method's own tens+digit guess, same convention as
    reconstruct_fdi_predictions - a wrong-quadrant tooth contributes whatever wrong number
    production would actually show, not the true one), then F1 is computed per arch and averaged
    across arches - unlike compute_method_comparison's macro F1, which averages per DIGIT CLASS
    across all teeth pooled together.

    AUC: per-tooth AUC needs one score per class; there's no equivalent per-arch multi-class
    ranking, so instead this asks a narrower, still useful question - does the model's own
    confidence signal predict whether the WHOLE arch will come back exactly right? Each arch's
    score is the minimum per-tooth probability the model assigned to its own (pre-gating) chosen
    digit across every tooth in that arch - an arch is only as confident as its least-confident
    tooth. AUC then measures how well that score separates exact-match arches from non-exact
    ones. NaN wherever a series has only one label value (e.g. every arch is - or isn't - an
    exact match), the same as compute_method_comparison's per-class AUC guard.
    """
    results = {}
    for series in ("lower", "upper", "pooled"):
        jaws = ("lower", "upper") if series == "pooled" else (series,)

        series_result = {}
        for key, label, pred_field, prob_field, tens_pred_field in METHODS:
            arch_true_sets, arch_pred_sets = {}, {}
            arch_confidence, arch_exact = {}, {}

            for jaw in jaws:
                d = per_jaw[jaw]
                for pred_digit, true_digit, tens_pred, fdi_number, image_name, prob_row in zip(
                        d[pred_field], d["y_true"], d[tens_pred_field], d["y_fdi_number"],
                        d["y_image_name"], d[prob_field]):
                    arch_key = (jaw, image_name)
                    arch_true_sets.setdefault(arch_key, set()).add(fdi_number)
                    arch_pred_sets.setdefault(arch_key, set()).add(tens_pred * 10 + (pred_digit + 1))

                    is_correct = pred_digit == true_digit and tens_pred == (fdi_number // 10)
                    arch_exact[arch_key] = arch_exact.get(arch_key, True) and is_correct
                    conf = prob_row[pred_digit]
                    arch_confidence[arch_key] = min(arch_confidence.get(arch_key, 1.0), conf)

            arch_keys = list(arch_true_sets.keys())
            n_arches = len(arch_keys)

            accuracy = sum(arch_exact[k] for k in arch_keys) / n_arches if n_arches else 0.0

            precision_per_arch, recall_per_arch, f1_per_arch = [], [], []
            for k in arch_keys:
                true_set, pred_set = arch_true_sets[k], arch_pred_sets[k]
                tp = len(true_set & pred_set)
                if tp == 0:
                    precision_per_arch.append(0.0)
                    recall_per_arch.append(0.0)
                    f1_per_arch.append(0.0)
                    continue
                precision = tp / len(pred_set)
                recall = tp / len(true_set)
                precision_per_arch.append(precision)
                recall_per_arch.append(recall)
                f1_per_arch.append(2 * precision * recall / (precision + recall))
            precision_avg = sum(precision_per_arch) / n_arches if n_arches else 0.0
            recall_avg = sum(recall_per_arch) / n_arches if n_arches else 0.0
            f1 = sum(f1_per_arch) / n_arches if n_arches else 0.0

            labels = [1 if arch_exact[k] else 0 for k in arch_keys]
            scores = [arch_confidence[k] for k in arch_keys]
            try:
                if len(set(labels)) < 2:
                    raise ValueError("only one class present in this series")
                auc = roc_auc_score(labels, scores)
            except ValueError:
                auc = float("nan")

            series_result[key] = {
                "label": label, "accuracy": accuracy,
                "precision": precision_avg, "recall": recall_avg, "f1": f1, "auc": auc,
            }
        results[series] = series_result
    return results


def compute_missing_teeth_metrics(per_jaw, pred_key="y_postproc", prob_field="prob_resnet",
                                   tens_pred_key="y_tens_pred_hungarian", full_arch_size=12):
    """Same per-arch metrics as compute_arch_level_metrics (Exact-Match Accuracy / sample-averaged
    Precision, Recall, F1 over FDI-number sets / min-confidence AUC), but bucketed by how many
    teeth are MISSING from the arch instead of by which method produced the prediction - answers
    "does this pipeline degrade as more teeth go missing" rather than "which method is best."

    Only evaluates one prediction source (default: production's actual ResNet+Hungarian output),
    not all of METHODS - splitting all five by missing-count as well would blow up the table for a
    question ("is the adopted method robust to missing teeth") that only needs the adopted one.

    missing_count for an arch = full_arch_size (12: the FDI last-digit 1-6 x 2 sides this pipeline
    covers - digits 7/8 aren't in the dataset at all, see the paper's Limitations section) minus
    however many teeth actually have a GT label in that arch. Returns {missing_count: {n_arches,
    accuracy, precision, recall, f1, auc}}, sorted by missing_count ascending - buckets with very
    few arches (check n_arches) should be read cautiously, not over-interpreted.
    """
    arch_true_sets, arch_pred_sets = {}, {}
    arch_confidence, arch_exact = {}, {}

    for jaw in ("lower", "upper"):
        d = per_jaw[jaw]
        for pred_digit, true_digit, tens_pred, fdi_number, image_name, prob_row in zip(
                d[pred_key], d["y_true"], d[tens_pred_key], d["y_fdi_number"],
                d["y_image_name"], d[prob_field]):
            arch_key = (jaw, image_name)
            arch_true_sets.setdefault(arch_key, set()).add(fdi_number)
            arch_pred_sets.setdefault(arch_key, set()).add(tens_pred * 10 + (pred_digit + 1))

            is_correct = pred_digit == true_digit and tens_pred == (fdi_number // 10)
            arch_exact[arch_key] = arch_exact.get(arch_key, True) and is_correct
            conf = prob_row[pred_digit]
            arch_confidence[arch_key] = min(arch_confidence.get(arch_key, 1.0), conf)

    buckets = {}
    for k, true_set in arch_true_sets.items():
        missing_count = max(full_arch_size - len(true_set), 0)
        buckets.setdefault(missing_count, []).append(k)

    results = {}
    for missing_count in sorted(buckets):
        keys = buckets[missing_count]
        n_arches = len(keys)

        accuracy = sum(arch_exact[k] for k in keys) / n_arches

        precision_per_arch, recall_per_arch, f1_per_arch = [], [], []
        for k in keys:
            true_set, pred_set = arch_true_sets[k], arch_pred_sets[k]
            tp = len(true_set & pred_set)
            if tp == 0:
                precision_per_arch.append(0.0)
                recall_per_arch.append(0.0)
                f1_per_arch.append(0.0)
                continue
            precision = tp / len(pred_set)
            recall = tp / len(true_set)
            precision_per_arch.append(precision)
            recall_per_arch.append(recall)
            f1_per_arch.append(2 * precision * recall / (precision + recall))

        labels = [1 if arch_exact[k] else 0 for k in keys]
        scores = [arch_confidence[k] for k in keys]
        try:
            if len(set(labels)) < 2:
                raise ValueError("only one class present in this bucket")
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = float("nan")

        results[missing_count] = {
            "n_arches": n_arches,
            "accuracy": accuracy,
            "precision": sum(precision_per_arch) / n_arches,
            "recall": sum(recall_per_arch) / n_arches,
            "f1": sum(f1_per_arch) / n_arches,
            "auc": auc,
        }
    return results


def compute_missing_vs_complete_metrics(per_jaw, pred_key="y_postproc", prob_field="prob_resnet",
                                         tens_pred_key="y_tens_pred_hungarian", full_arch_size=12):
    """Same per-arch metrics as compute_missing_teeth_metrics, collapsed into just two groups -
    "complete" (0 missing teeth) vs "missing" (>=1 missing) - instead of one bucket per exact
    missing count. Meant as a robustness comparison that stays legible once other models/methods
    are added as extra bars later (see plot_missing_vs_complete) - the finer per-count breakdown
    gets cluttered fast once more than one series is on the same chart, and several of its higher
    counts have too few arches to compare across models anyway (see tab:missing_teeth's own
    caveat about the 3+ buckets).

    Recomputes from per_jaw directly (not by merging compute_missing_teeth_metrics's per-count
    output) so precision/recall/f1 stay sample-averaged over the arches actually in each of the
    two groups, not an average-of-averages across missing counts.
    """
    arch_true_sets, arch_pred_sets = {}, {}
    arch_confidence, arch_exact = {}, {}

    for jaw in ("lower", "upper"):
        d = per_jaw[jaw]
        for pred_digit, true_digit, tens_pred, fdi_number, image_name, prob_row in zip(
                d[pred_key], d["y_true"], d[tens_pred_key], d["y_fdi_number"],
                d["y_image_name"], d[prob_field]):
            arch_key = (jaw, image_name)
            arch_true_sets.setdefault(arch_key, set()).add(fdi_number)
            arch_pred_sets.setdefault(arch_key, set()).add(tens_pred * 10 + (pred_digit + 1))

            is_correct = pred_digit == true_digit and tens_pred == (fdi_number // 10)
            arch_exact[arch_key] = arch_exact.get(arch_key, True) and is_correct
            conf = prob_row[pred_digit]
            arch_confidence[arch_key] = min(arch_confidence.get(arch_key, 1.0), conf)

    groups = {"complete": [], "missing": []}
    for k, true_set in arch_true_sets.items():
        missing_count = max(full_arch_size - len(true_set), 0)
        groups["complete" if missing_count == 0 else "missing"].append(k)

    results = {}
    for group_name in ("complete", "missing"):
        keys = groups[group_name]
        n_arches = len(keys)
        if n_arches == 0:
            results[group_name] = {"n_arches": 0, "accuracy": 0.0, "precision": 0.0,
                                    "recall": 0.0, "f1": 0.0, "auc": float("nan")}
            continue

        accuracy = sum(arch_exact[k] for k in keys) / n_arches

        precision_per_arch, recall_per_arch, f1_per_arch = [], [], []
        for k in keys:
            true_set, pred_set = arch_true_sets[k], arch_pred_sets[k]
            tp = len(true_set & pred_set)
            if tp == 0:
                precision_per_arch.append(0.0)
                recall_per_arch.append(0.0)
                f1_per_arch.append(0.0)
                continue
            precision = tp / len(pred_set)
            recall = tp / len(true_set)
            precision_per_arch.append(precision)
            recall_per_arch.append(recall)
            f1_per_arch.append(2 * precision * recall / (precision + recall))

        labels = [1 if arch_exact[k] else 0 for k in keys]
        scores = [arch_confidence[k] for k in keys]
        try:
            if len(set(labels)) < 2:
                raise ValueError("only one class present in this group")
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = float("nan")

        results[group_name] = {
            "n_arches": n_arches,
            "accuracy": accuracy,
            "precision": sum(precision_per_arch) / n_arches,
            "recall": sum(recall_per_arch) / n_arches,
            "f1": sum(f1_per_arch) / n_arches,
            "auc": auc,
        }
    return results


def compute_arch_metrics_by_complexity(per_jaw, complexity_by_image, pred_key="y_postproc",
                                        prob_field="prob_resnet", tens_pred_key="y_tens_pred_hungarian"):
    """Same per-arch metrics as compute_missing_vs_complete_metrics, bucketed by each arch's
    ground-truth Arch Complexity class (1/2/3, from AIHub's metadata.json - see
    Transformer/complexity/build_dataset.py's load_complexity_labels) instead of missing-tooth status -
    answers "does whole-arch exact-match performance hold up as malocclusion severity increases"
    rather than "...as teeth go missing." Returns {complexity_class: metrics}, keyed 1/2/3 so it
    slots into plot_grouped_model_comparison the same way compute_missing_vs_complete_metrics's
    output does."""
    arch_true_sets, arch_pred_sets = {}, {}
    arch_confidence, arch_exact, arch_complexity = {}, {}, {}

    for jaw in ("lower", "upper"):
        d = per_jaw[jaw]
        for pred_digit, true_digit, tens_pred, fdi_number, image_name, prob_row in zip(
                d[pred_key], d["y_true"], d[tens_pred_key], d["y_fdi_number"],
                d["y_image_name"], d[prob_field]):
            arch_key = (jaw, image_name)
            arch_true_sets.setdefault(arch_key, set()).add(fdi_number)
            arch_pred_sets.setdefault(arch_key, set()).add(tens_pred * 10 + (pred_digit + 1))

            is_correct = pred_digit == true_digit and tens_pred == (fdi_number // 10)
            arch_exact[arch_key] = arch_exact.get(arch_key, True) and is_correct
            conf = prob_row[pred_digit]
            arch_confidence[arch_key] = min(arch_confidence.get(arch_key, 1.0), conf)
            if arch_key not in arch_complexity:
                arch_complexity[arch_key] = complexity_by_image.get(image_name)

    buckets = {}
    for k in arch_true_sets:
        c = arch_complexity.get(k)
        if c is None:
            continue
        buckets.setdefault(c, []).append(k)

    results = {}
    for complexity_class in sorted(buckets):
        keys = buckets[complexity_class]
        n_arches = len(keys)

        accuracy = sum(arch_exact[k] for k in keys) / n_arches

        precision_per_arch, recall_per_arch, f1_per_arch = [], [], []
        for k in keys:
            true_set, pred_set = arch_true_sets[k], arch_pred_sets[k]
            tp = len(true_set & pred_set)
            if tp == 0:
                precision_per_arch.append(0.0)
                recall_per_arch.append(0.0)
                f1_per_arch.append(0.0)
                continue
            precision = tp / len(pred_set)
            recall = tp / len(true_set)
            precision_per_arch.append(precision)
            recall_per_arch.append(recall)
            f1_per_arch.append(2 * precision * recall / (precision + recall))

        labels = [1 if arch_exact[k] else 0 for k in keys]
        scores = [arch_confidence[k] for k in keys]
        try:
            if len(set(labels)) < 2:
                raise ValueError("only one class present in this bucket")
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = float("nan")

        results[complexity_class] = {
            "n_arches": n_arches,
            "accuracy": accuracy,
            "precision": sum(precision_per_arch) / n_arches,
            "recall": sum(recall_per_arch) / n_arches,
            "f1": sum(f1_per_arch) / n_arches,
            "auc": auc,
        }
    return results


def plot_grouped_model_comparison(results_by_model, group_order, group_labels, title, out_path):
    """5 stacked panels (Precision/Recall/F1/Accuracy/AUC), each a grouped bar chart: one group
    per bucket in group_order (e.g. missing-tooth status from compute_missing_vs_complete_metrics,
    or Arch Complexity class from compute_arch_metrics_by_complexity), one bar per model in
    results_by_model. Takes {model_label: {group_key: metrics}} so adding another model later
    (e.g. the comparison/ghorbani or comparison/nguyen reproductions, once they have their own
    arch-level predictions to bucket the same way) is just adding another key to that dict and
    replotting - a small, fixed set of groups on the x-axis is what keeps this legible once more
    than one model's bars share the same chart, unlike a finer per-count/per-value breakdown."""
    metric_keys = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1 (Sample-avg)"),
                   ("accuracy", "Accuracy (Exact-Match)"), ("auc", "AUC")]
    model_keys = list(results_by_model.keys())
    palette = [BLUE, ORANGE, PURPLE, TEAL, PINK]

    x = list(range(len(group_order)))
    n_series = len(model_keys)
    bar_width = 0.8 / max(n_series, 1)

    fig, axes = plt.subplots(len(metric_keys), 1, figsize=(7.5, 4.2 * len(metric_keys)), facecolor=SURFACE)
    for ax, (metric_key, metric_label) in zip(axes, metric_keys):
        for i, model_key in enumerate(model_keys):
            offsets = [xi + (i - (n_series - 1) / 2) * bar_width for xi in x]
            values = [results_by_model[model_key][g][metric_key] for g in group_order]
            bars = ax.bar(offsets, values, width=bar_width, color=palette[i % len(palette)],
                           label=model_key, zorder=3)
            annotate_bars(ax, bars)

        ax.set_xticks(x)
        ax.set_xticklabels(group_labels)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(metric_label, color=INK)
        ax.legend(frameon=False, labelcolor=INK, fontsize=8)
        style_axes(ax)

    fig.suptitle(title, color=INK, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close()


def plot_method_comparison(comparison, title, out_path):
    """Three stacked panels (Accuracy / F1 / AUC), each a grouped bar chart: one group per
    series (Lower / Upper / Pooled jaw), one bar per method (METHODS)."""
    series_order = ["lower", "upper", "pooled"]
    metric_keys = [("accuracy", "Accuracy"), ("f1", "F1 (Macro)"), ("auc", "AUC (Macro, OvR)")]
    method_keys = [key for key, *_ in METHODS]

    x = list(range(len(series_order)))
    n_series = len(method_keys)
    bar_width = 0.8 / n_series

    fig, axes = plt.subplots(len(metric_keys), 1, figsize=(11, 5 * len(metric_keys)), facecolor=SURFACE)
    for ax, (metric_key, metric_label) in zip(axes, metric_keys):
        for i, method_key in enumerate(method_keys):
            offsets = [xi + (i - (n_series - 1) / 2) * bar_width for xi in x]
            values = [comparison[s][method_key][metric_key] for s in series_order]
            method_label = comparison[series_order[0]][method_key]["label"]
            bars = ax.bar(offsets, values, width=bar_width, color=METHOD_COLORS[method_key],
                           label=method_label, zorder=3)
            annotate_bars(ax, bars)

        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in series_order])
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(metric_label, color=INK)
        ax.legend(frameon=False, labelcolor=INK, fontsize=8)
        style_axes(ax)

    fig.suptitle(title, color=INK, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close()


def reconstruct_fdi_predictions(per_jaw, pred_key="y_postproc", tens_pred_key="y_tens_pred_hungarian"):
    """Reconstructs a full predicted FDI number as predicted_tens*10 + predicted_digit - tens
    from tens_pred_key (production's actual tens guess for this prediction source - see METHODS
    for why resnet_hungarian gets its own field), NOT the ground truth, so a wrong-quadrant tooth
    reconstructs to whatever wrong FDI number production would actually have shown (e.g. true 41
    landing on predicted-tens 3 becomes "31", genuinely confusable with a true 31 tooth) instead
    of always resolving to the true quadrant. Pooled across both jaws. Shared by
    compute_tooth_number_metrics_by_number and plot_tooth_number_confusion_by_fdi so the
    reconstruction logic (and which prediction source it's built from) can't drift between the
    two."""
    all_true_fdi, all_pred_fdi = [], []
    for jaw in ("lower", "upper"):
        d = per_jaw[jaw]
        for pred_digit, tens_pred, fdi_number in zip(d[pred_key], d[tens_pred_key], d["y_fdi_number"]):
            all_true_fdi.append(fdi_number)
            all_pred_fdi.append(tens_pred * 10 + (pred_digit + 1))
    return all_true_fdi, all_pred_fdi


def plot_tooth_number_confusion_by_fdi(per_jaw, title, out_path, pred_key="y_postproc",
                                        tens_pred_key="y_tens_pred_hungarian"):
    """Full multiclass confusion matrix over actual FDI numbers (11-46, digits 1-6 only - the
    6-class model was never trained on the 7/8 wisdom-tooth digits) for the pipeline's final
    prediction, reconstructed the same way compute_tooth_number_metrics_by_number's per-number
    breakdown is. Unlike that per-number accuracy/precision/recall bar chart, this shows exactly
    which OTHER number a miss gets confused with (e.g. is 14 mistaken for 13 or for 15?),
    not just that 14's own hit-rate dropped.

    Also saves a row-normalized ("{out_path.stem}_pct.png") companion: raw counts are dominated
    by the large diagonal of common numbers, which buries how a rare number's few instances
    split across error types - normalizing each true-label row to sum to 100% puts every FDI
    number on the same scale regardless of how many test instances it had."""
    fdi_labels = [n for n in ALL_FDI_NUMBERS if 1 <= n % 10 <= 6]
    all_true_fdi, all_pred_fdi = reconstruct_fdi_predictions(per_jaw, pred_key, tens_pred_key)

    cm = confusion_matrix(all_true_fdi, all_pred_fdi, labels=fdi_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(n) for n in fdi_labels])
    # 24x24 is a lot of cells for a 13.8x13in figure (~0.55in/cell) - sklearn's own auto-sized
    # annotation font shrinks further as class count grows, so it's bumped explicitly here
    # (9pt keeps 2-3 digit counts from colliding at this cell size; ticks/title follow suit).
    fig, ax = plt.subplots(figsize=(17, 16), facecolor=SURFACE)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True, xticks_rotation=90, values_format="d",
              text_kw={"fontsize": 11})
    ax.set_title(title, color=INK, fontsize=16)
    ax.set_xlabel(ax.get_xlabel(), fontsize=14)
    ax.set_ylabel(ax.get_ylabel(), fontsize=14)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close()

    cm_pct = confusion_matrix(all_true_fdi, all_pred_fdi, labels=fdi_labels, normalize="true") * 100
    disp_pct = ConfusionMatrixDisplay(confusion_matrix=cm_pct, display_labels=[str(n) for n in fdi_labels])
    fig, ax = plt.subplots(figsize=(17, 16), facecolor=SURFACE)
    # A linear 0-100 color scale makes the small off-diagonal error percentages (often <1%)
    # indistinguishable from 0 next to the ~100% diagonal. PowerNorm(gamma<1) compresses the
    # high end and stretches the low end instead, so a cell at 0.1-0.2% still gets visible color
    # rather than reading as pure white - gamma=0.35 chosen empirically to keep small-but-real
    # error rates visible without also making 0% cells look colored.
    disp_pct.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True, xticks_rotation=90, values_format=".1f",
                   im_kw={"norm": PowerNorm(gamma=0.35, vmin=0, vmax=100)}, text_kw={"fontsize": 11})
    ax.set_title(f"{title} (%)", color=INK, fontsize=16)
    ax.set_xlabel(ax.get_xlabel(), fontsize=14)
    ax.set_ylabel(ax.get_ylabel(), fontsize=14)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    pct_path = out_path.with_name(f"{out_path.stem}_pct{out_path.suffix}")
    plt.savefig(pct_path, dpi=150, facecolor=SURFACE)
    plt.close()


def compute_tooth_number_metrics_by_number(per_jaw):
    """Accuracy/precision/recall of the final (Hungarian-corrected where applicable) prediction,
    grouped by the tooth's true full FDI number rather than just its last digit - e.g. is 48
    (wisdom tooth) harder to get right than 41? Uses reconstruct_fdi_predictions() (see there for
    how the full number is rebuilt), so this is a faithful multiclass precision/recall over the
    tooth positions the model can predict. F1 is omitted here - with exactly one predicted label
    per instance, per-class recall already equals per-class hit-rate ("accuracy"), and F1 would
    just restate precision/recall."""
    all_true_fdi, all_pred_fdi = reconstruct_fdi_predictions(per_jaw)

    present = sorted(set(all_true_fdi) | set(all_pred_fdi))
    precision, recall, _, _ = precision_recall_fscore_support(
        all_true_fdi, all_pred_fdi, labels=present, average=None, zero_division=0
    )

    # Per-class recall == per-class hit-rate here (single predicted label per instance), so
    # "accuracy" and "recall" coincide - kept as separate keys for symmetry with other charts.
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
    print("Tooth number - is the predicted FDI digit the true one?")
    print("=" * 70)
    per_jaw = evaluate_tooth_number()

    def tens_accuracy_of(tens_pred_key):
        acc = {}
        for jaw in ("lower", "upper"):
            tp = per_jaw[jaw][tens_pred_key]
            fn = per_jaw[jaw]["y_fdi_number"]
            correct = [t == (n // 10) for t, n in zip(tp, fn)]
            acc[jaw] = sum(correct) / len(correct) if correct else 0.0
        all_tp = per_jaw["lower"][tens_pred_key] + per_jaw["upper"][tens_pred_key]
        all_fn = per_jaw["lower"]["y_fdi_number"] + per_jaw["upper"]["y_fdi_number"]
        all_correct = [t == (n // 10) for t, n in zip(all_tp, all_fn)]
        acc["pooled"] = sum(all_correct) / len(all_correct) if all_correct else 0.0
        return acc

    tens_accuracy = tens_accuracy_of("y_tens_pred")
    tens_accuracy_hungarian = tens_accuracy_of("y_tens_pred_hungarian")
    print("FDI tens-digit (quadrant) accuracy - geometry-only guess vs. ResNet+Hungarian's own "
          "(digit-pairing-corrected, see evaluate_tooth_number):")
    for series in ("lower", "upper", "pooled"):
        print(f"  {series}: geometry-only={tens_accuracy[series]:.4f}  "
              f"resnet_hungarian={tens_accuracy_hungarian[series]:.4f}")

    # ResNet-only baseline vs the pipeline's final ResNet+Hungarian output, plotted and scored
    # separately so the post-process's effect is directly visible (both predictions were already
    # collected per-tooth in evaluate_tooth_number(), just not both surfaced before).
    plot_tooth_number_confusion(
        per_jaw, "y_resnet", "ResNet-only", RESULTS_DIR / "tooth_number_confusion_resnet_{jaw}.png"
    )
    plot_tooth_number_confusion(
        per_jaw, "y_postproc", "ResNet+Hungarian", RESULTS_DIR / "tooth_number_confusion_hungarian_{jaw}.png",
        tens_pred_key="y_tens_pred_hungarian"
    )

    resnet_metrics = compute_tooth_number_metrics(per_jaw, "y_resnet")
    hungarian_metrics = compute_tooth_number_metrics(per_jaw, "y_postproc", tens_pred_key="y_tens_pred_hungarian")
    print(f"ResNet-only:      F1={resnet_metrics['f1']:.4f}  Accuracy={resnet_metrics['accuracy']:.4f}")
    print(f"ResNet+Hungarian: F1={hungarian_metrics['f1']:.4f}  Accuracy={hungarian_metrics['accuracy']:.4f}")
    plot_metric_bars(
        resnet_metrics,
        "Tooth Number Performance - ResNet-only (Test Set)",
        RESULTS_DIR / "tooth_number_metrics_resnet.png"
    )
    plot_metric_bars(
        hungarian_metrics,
        "Tooth Number Performance - ResNet+Hungarian (Test Set)",
        RESULTS_DIR / "tooth_number_metrics_hungarian.png"
    )

    number_metrics_by_number = compute_tooth_number_metrics_by_number(per_jaw)
    plot_metrics_by_number(
        number_metrics_by_number,
        "Tooth Number by Tooth Number (ResNet+Hungarian, Test Set)",
        RESULTS_DIR / "tooth_number_by_number.png"
    )
    plot_tooth_number_confusion_by_fdi(
        per_jaw,
        "Tooth Number Confusion Matrix by FDI Number (ResNet+Hungarian, Test Set)",
        RESULTS_DIR / "tooth_number_confusion_by_fdi.png"
    )

    print("\n" + "=" * 70)
    print("Method comparison - ResNet-only vs ResNet+ViT vs three post-process families")
    print("=" * 70)
    method_comparison = compute_method_comparison(per_jaw)
    for series in ("lower", "upper", "pooled"):
        for key, label, *_ in METHODS:
            m = method_comparison[series][key]
            print(f"{series:6s} {label:20s} acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  auc={m['auc']:.4f}")
    plot_method_comparison(
        method_comparison,
        "Tooth Number Method Comparison - Accuracy / F1 / AUC (Test Set)",
        RESULTS_DIR / "tooth_number_method_comparison.png"
    )

    print("\n" + "=" * 70)
    print("Method comparison (ARCH-LEVEL) - Exact-Match Accuracy / Sample F1 / Confidence AUC")
    print("=" * 70)
    arch_method_comparison = compute_arch_level_metrics(per_jaw)
    for series in ("lower", "upper", "pooled"):
        for key, label, *_ in METHODS:
            m = arch_method_comparison[series][key]
            print(f"{series:6s} {label:20s} acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  auc={m['auc']:.4f}")
    plot_method_comparison(
        arch_method_comparison,
        "Tooth Number Method Comparison - Arch-Level Accuracy / F1 / AUC (Test Set)",
        RESULTS_DIR / "tooth_number_method_comparison_arch_level.png"
    )

    print("\n" + "=" * 70)
    print("ResNet+Hungarian arch-level metrics by missing-tooth count")
    print("=" * 70)
    missing_teeth_metrics = compute_missing_teeth_metrics(per_jaw)
    for missing_count, m in missing_teeth_metrics.items():
        print(f"missing={missing_count:2d}  n_arches={m['n_arches']:5d}  "
              f"precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
              f"f1={m['f1']:.4f}  acc={m['accuracy']:.4f}  auc={m['auc']:.4f}")

    print("\n" + "=" * 70)
    print("ResNet+Hungarian arch-level metrics: complete vs missing-tooth arches")
    print("=" * 70)
    missing_vs_complete = compute_missing_vs_complete_metrics(per_jaw)
    for group_name, m in missing_vs_complete.items():
        print(f"{group_name:10s} n_arches={m['n_arches']:5d}  "
              f"precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
              f"f1={m['f1']:.4f}  acc={m['accuracy']:.4f}  auc={m['auc']:.4f}")
    # Keyed by model label rather than a bare metrics dict so a later run can add
    # comparison/ghorbani's or comparison/nguyen's own {group: metrics} under their own key and
    # replot the same chart with multiple models' bars side by side (see
    # plot_grouped_model_comparison's docstring).
    missing_vs_complete_by_model = {"CHAI (ResNet+Hungarian)": missing_vs_complete}
    plot_grouped_model_comparison(
        missing_vs_complete_by_model,
        group_order=["complete", "missing"],
        group_labels=["Complete\n(0 missing)", "Missing\n(≥ 1)"],
        title="Arch-Level Robustness: Complete vs Missing-Tooth Arches (Test Set)",
        out_path=RESULTS_DIR / "tooth_number_missing_vs_complete.png"
    )

    print("\n" + "=" * 70)
    print("ResNet+Hungarian arch-level metrics by Arch Complexity class")
    print("=" * 70)
    complexity_build = load_module(
        "vit_complexity_build_eval_2", PROJECT_ROOT / "Transformer" / "complexity" / "build_dataset.py"
    )
    complexity_by_image = complexity_build.load_complexity_labels(DATASET_DIR, "test")
    arch_by_complexity = compute_arch_metrics_by_complexity(per_jaw, complexity_by_image)
    complexity_class_labels = {1: "Class I", 2: "Class II", 3: "Class III"}
    for complexity_class, m in arch_by_complexity.items():
        print(f"{complexity_class_labels.get(complexity_class, complexity_class):10s} "
              f"n_arches={m['n_arches']:5d}  precision={m['precision']:.4f}  "
              f"recall={m['recall']:.4f}  f1={m['f1']:.4f}  acc={m['accuracy']:.4f}  auc={m['auc']:.4f}")

    arch_by_complexity_by_model = {"CHAI (ResNet+Hungarian)": arch_by_complexity}
    plot_grouped_model_comparison(
        arch_by_complexity_by_model,
        group_order=[2, 3],
        group_labels=["Class II", "Class III"],
        title="Arch-Level Robustness by Arch Complexity Class (Test Set)",
        out_path=RESULTS_DIR / "tooth_number_arch_level_by_complexity.png"
    )

    per_tooth_keys = ("y_true", "y_resnet", "y_vit", "y_postproc", "y_monodp", "y_greedy",
                       "prob_resnet", "prob_vit", "y_tens_pred",
                       "y_tens_pred_hungarian", "y_fdi_number", "y_image_name")
    summary["tooth_number"] = {
        "tens_digit_accuracy": {"geometry_only": tens_accuracy, "resnet_hungarian": tens_accuracy_hungarian},
        "metrics": {"resnet_only": resnet_metrics, "resnet_hungarian": hungarian_metrics},
        "method_comparison": method_comparison,
        "arch_level_method_comparison": arch_method_comparison,
        "missing_teeth_metrics": missing_teeth_metrics,
        "missing_vs_complete_by_model": missing_vs_complete_by_model,
        "arch_level_by_complexity_by_model": arch_by_complexity_by_model,
        "metrics_by_number": number_metrics_by_number,
        "per_jaw": {
            jaw: {k: v for k, v in d.items() if k not in per_tooth_keys}
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
