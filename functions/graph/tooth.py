"""
Evaluates the tooth-number pipeline's final prediction (is the predicted FDI last digit the
true one?) on the held-out test split (dataset/test/, never touched by training or validation).

Reuses the project's existing pipelines instead of reimplementing them:
functions/features/main.py (GT feature CSV extraction), ViT/arch/build_dataset.py (ResNet
crop/feature/probability extraction per arch), and ResNet/tooth/postprocess.py (the per-quadrant
Hungarian duplicate-resolution post-process that replaced the ViT arch-transformer refinement
step - see that module's docstring for why).

Leads with macro F1 rather than accuracy: with the last-digit classes this imbalanced (incisors
vastly outnumber missing/rare positions), accuracy alone can look flat even as rare-class
performance moves - F1 is the metric worth watching first.

Usage:
    .venv/Scripts/python.exe functions/graph/tooth.py
"""
import csv as csv_module
import importlib.util
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

CSV_DIR = PROJECT_ROOT / "ResNet" / "tooth" / "csv"
RESNET_MODEL_DIR = PROJECT_ROOT / "ResNet" / "tooth" / "model"
ARCH_MODEL_DIR = PROJECT_ROOT / "ViT" / "arch" / "model"

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
    """Replicates production's actual (geometry-only) FDI tens-digit assignment - js/geometry.js
    computeToothMeta/computeQuadrantTens: PCA over every detected tooth centroid in the arch,
    then each tooth's own rotated-x sign decides its quadrant. The last digit is the only thing
    the model predicts; the tens digit has always come from this geometric step, both in
    production and, silently, in every eval chart in this module up to now - which used the
    *ground-truth* tens digit instead, i.e. assumed this geometric step is always correct. It
    isn't: a missing tooth skews the PCA-estimated midline, and the tooth most exposed to that
    skew is exactly the one closest to it (a digit-"1" tooth) - see project notes for a live
    example (a real 41 and 31 both reading as "31"). This function exists so that failure mode
    is actually visible in these graphs instead of being invisibly assumed away.

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
    """Runs every method being compared over the test set in one pass (one ResNet forward pass
    per jaw is the expensive part - everything downstream of `prob_vec` is cheap, so it's not
    worth splitting into separate eval functions per method):

      - y_resnet    : ResNet-only, independent per-tooth argmax (today's fallback whenever a
                      jaw has no arch weights).
      - y_vit       : ResNet + the ViT arch-transformer refinement this pipeline used to run in
                      production (see ViT/arch/model.py) - kept only for this comparison, not
                      used anywhere else anymore (see ResNet/tooth/postprocess.py's docstring
                      for why it was retired in favor of the Hungarian post-process below).
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
    features_main = load_module("features_main", PROJECT_ROOT / "functions" / "features" / "main.py")
    arch_build = load_module("arch_build_dataset", PROJECT_ROOT / "ViT" / "arch" / "build_dataset.py")
    postprocess = load_module("resnet_tooth_postprocess", PROJECT_ROOT / "ResNet" / "tooth" / "postprocess.py")
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
        # Production's actual (geometry-only, occasionally wrong) tens-digit guess - used below
        # both to gate correctness (gate_by_tens) and, faithfully to production, as the grouping
        # key the quadrant-dedup post-processes (Hungarian etc.) actually see - not the GT tens,
        # which would hand them cleaner groups than production ever gets.
        tens_pred_map = compute_tens_predictions(str(DATASET_DIR), jaw, "test")

        arch_weights = ARCH_MODEL_DIR / f"{jaw}_best.pth"
        arch_model = None
        if arch_weights.exists():
            print(f"Loading ViT arch transformer for {jaw} (comparison only, not production)...")
            arch_model = ArchToothTransformer(num_classes=6)
            arch_model.load_state_dict(torch.load(arch_weights, map_location=device))
            arch_model.to(device)
            arch_model.eval()
        else:
            print(f"No ViT arch transformer for {jaw} - ResNet+ViT column will fall back to ResNet-only.")

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
        fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
        ax.set_title(f"Tooth Number Confusion Matrix - {jaw.capitalize()} Jaw ({label})", color=INK)
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
                "f1": f1_score(y_true, y_pred_gated, labels=list(range(6)), average="macro", zero_division=0),
                "auc": auc,
            }
        results[series] = series_result
    return results


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
    not just that 14's own hit-rate dropped."""
    fdi_labels = [n for n in ALL_FDI_NUMBERS if 1 <= n % 10 <= 6]
    all_true_fdi, all_pred_fdi = reconstruct_fdi_predictions(per_jaw, pred_key, tens_pred_key)

    cm = confusion_matrix(all_true_fdi, all_pred_fdi, labels=fdi_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(n) for n in fdi_labels])
    fig, ax = plt.subplots(figsize=(13, 13), facecolor=SURFACE)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False, xticks_rotation=90, values_format="d")
    ax.set_title(title, color=INK)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=SURFACE)
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

    per_tooth_keys = ("y_true", "y_resnet", "y_vit", "y_postproc", "y_monodp", "y_greedy",
                       "prob_resnet", "prob_vit", "y_tens_pred",
                       "y_tens_pred_hungarian", "y_fdi_number", "y_image_name")
    summary["tooth_number"] = {
        "tens_digit_accuracy": {"geometry_only": tens_accuracy, "resnet_hungarian": tens_accuracy_hungarian},
        "metrics": {"resnet_only": resnet_metrics, "resnet_hungarian": hungarian_metrics},
        "method_comparison": method_comparison,
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
