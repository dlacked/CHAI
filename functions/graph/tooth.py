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
# (key, display label, per-tooth hard-label field, per-tooth probability field). AUC for the
# three post-process methods is scored against `prob_resnet` (see compute_method_comparison's
# docstring for why that's correct, not an oversight).
METHODS = [
    ("resnet_only", "ResNet-only", "y_resnet", "prob_resnet"),
    ("resnet_vit", "ResNet+ViT", "y_vit", "prob_vit"),
    ("resnet_hungarian", "ResNet+Hungarian", "y_postproc", "prob_resnet"),
    ("resnet_monodp", "ResNet+Monotonic-DP", "y_monodp", "prob_resnet"),
    ("resnet_greedy", "ResNet+Greedy", "y_greedy", "prob_resnet"),
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
      - y_postproc  : ResNet + per-quadrant Hungarian duplicate resolution - today's actual
                      production pipeline (server.py /tooth_predict).
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

                # Quadrant grouping for every eval-side post-process uses the GT FDI tens digit
                # (contiguous runs of teeth sharing a quadrant) - equivalent to production's
                # geometry-derived `mirror` flag (server.py /tooth_predict), just sourced from
                # GT since that's already on hand here and this is scoring, not inference.
                tens_keys = [n // 10 for n in fdi_numbers]
                y_postproc.extend(postprocess.resolve_quadrant_duplicates(probs_np, tens_keys).tolist())
                y_monodp.extend(postprocess.resolve_quadrant_monotonic(probs_np, tens_keys).tolist())
                y_greedy.extend(postprocess.resolve_quadrant_greedy(probs_np, tens_keys).tolist())

        per_jaw[jaw] = {
            "y_true": y_true,
            "y_resnet": y_resnet, "y_vit": y_vit, "y_postproc": y_postproc,
            "y_monodp": y_monodp, "y_greedy": y_greedy,
            "prob_resnet": prob_resnet, "prob_vit": prob_vit,
            "y_fdi_number": y_fdi_number, "y_image_name": y_image_name,
            "n_arches": len(sequences), "n_teeth": len(y_true),
        }
        print(f"{jaw}: {len(sequences)} arches, {len(y_true)} teeth.")

    return per_jaw


def plot_tooth_number_confusion(per_jaw, pred_key, label, out_path_template):
    """Plots one confusion matrix per jaw for a single prediction source (pred_key is
    'y_resnet' for the ResNet-only baseline or 'y_postproc' for the pipeline's final,
    Hungarian-corrected output - both are already collected per-tooth in
    evaluate_tooth_number(), just not both plotted before). Called twice from main() so the two
    are directly comparable side by side."""
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
    """F1 (macro, primary)/accuracy/precision/recall over the 6 last-digit classes for one
    prediction source (pred_key: 'y_resnet' for the ResNet-only baseline, 'y_postproc' for the
    pipeline's final Hungarian-corrected output) against the GT last digit, pooled across both
    jaws."""
    all_true, all_pred = [], []
    for jaw in ("lower", "upper"):
        d = per_jaw[jaw]
        all_true += d["y_true"]
        all_pred += d[pred_key]

    return {
        "f1": f1_score(all_true, all_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(all_true, all_pred),
        "precision": precision_score(all_true, all_pred, average="macro", zero_division=0),
        "recall": recall_score(all_true, all_pred, average="macro", zero_division=0),
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
    """
    results = {}
    for series in ("lower", "upper", "pooled"):
        if series == "pooled":
            y_true = per_jaw["lower"]["y_true"] + per_jaw["upper"]["y_true"]
        else:
            y_true = per_jaw[series]["y_true"]

        series_result = {}
        for key, label, pred_field, prob_field in METHODS:
            if series == "pooled":
                y_pred = per_jaw["lower"][pred_field] + per_jaw["upper"][pred_field]
                y_prob = per_jaw["lower"][prob_field] + per_jaw["upper"][prob_field]
            else:
                y_pred = per_jaw[series][pred_field]
                y_prob = per_jaw[series][prob_field]

            try:
                auc = roc_auc_score(y_true, np.array(y_prob), multi_class="ovr",
                                     average="macro", labels=list(range(6)))
            except ValueError:
                auc = float("nan")  # a class is entirely absent from this series - can't rank it

            series_result[key] = {
                "label": label,
                "accuracy": accuracy_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
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


def reconstruct_fdi_predictions(per_jaw, pred_key="y_postproc"):
    """Reconstructs a full predicted FDI number as known_tens*10 + predicted_digit (tens comes
    from the tooth's own geometry, exactly like the live site's computeQuadrantTens - the model
    only ever predicts the last digit), pooled across both jaws. Shared by
    compute_tooth_number_metrics_by_number and plot_tooth_number_confusion_by_fdi so the
    reconstruction logic (and which prediction source it's built from) can't drift between the
    two."""
    all_true_fdi, all_pred_fdi = [], []
    for jaw in ("lower", "upper"):
        d = per_jaw[jaw]
        for pred_digit, fdi_number in zip(d[pred_key], d["y_fdi_number"]):
            tens = fdi_number // 10
            all_true_fdi.append(fdi_number)
            all_pred_fdi.append(tens * 10 + (pred_digit + 1))
    return all_true_fdi, all_pred_fdi


def plot_tooth_number_confusion_by_fdi(per_jaw, title, out_path, pred_key="y_postproc"):
    """Full multiclass confusion matrix over actual FDI numbers (11-46, digits 1-6 only - the
    6-class model was never trained on the 7/8 wisdom-tooth digits) for the pipeline's final
    prediction, reconstructed the same way compute_tooth_number_metrics_by_number's per-number
    breakdown is. Unlike that per-number accuracy/precision/recall bar chart, this shows exactly
    which OTHER number a miss gets confused with (e.g. is 14 mistaken for 13 or for 15?),
    not just that 14's own hit-rate dropped."""
    fdi_labels = [n for n in ALL_FDI_NUMBERS if 1 <= n % 10 <= 6]
    all_true_fdi, all_pred_fdi = reconstruct_fdi_predictions(per_jaw, pred_key)

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

    # ResNet-only baseline vs the pipeline's final ResNet+Hungarian output, plotted and scored
    # separately so the post-process's effect is directly visible (both predictions were already
    # collected per-tooth in evaluate_tooth_number(), just not both surfaced before).
    plot_tooth_number_confusion(
        per_jaw, "y_resnet", "ResNet-only", RESULTS_DIR / "tooth_number_confusion_resnet_{jaw}.png"
    )
    plot_tooth_number_confusion(
        per_jaw, "y_postproc", "ResNet+Hungarian", RESULTS_DIR / "tooth_number_confusion_hungarian_{jaw}.png"
    )

    resnet_metrics = compute_tooth_number_metrics(per_jaw, "y_resnet")
    hungarian_metrics = compute_tooth_number_metrics(per_jaw, "y_postproc")
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
                       "prob_resnet", "prob_vit", "y_fdi_number", "y_image_name")
    summary["tooth_number"] = {
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
