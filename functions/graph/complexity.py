"""
Evaluates the ViT/complexity arch-level Arch Complexity model (Class I/II/III,
predicted from per-tooth geometry alone - see ViT/complexity/model.py) on the held-out test
split (dataset/test/, never touched by training or validation).

Reuses ViT/complexity/{model,dataset}.py and the {jaw}_test.pt caches already built by
ViT/complexity/build_dataset.py instead of reimplementing the data pipeline.

Usage:
    .venv/Scripts/python.exe functions/graph/complexity.py
"""
import json
import importlib.util
from collections import defaultdict
from pathlib import Path

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

GRAPH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GRAPH_DIR.parent.parent  # .../CHAI/CHAI
RESULTS_DIR = GRAPH_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COMPLEXITY_DIR = PROJECT_ROOT / "ViT" / "complexity"
CACHE_DIR = COMPLEXITY_DIR / "cache"
MODEL_DIR = COMPLEXITY_DIR / "model"

DISPLAY_LABELS = ["Class I", "Class II", "Class III"]  # Arch Complexity display text

# Raw complexity values from metadata.json (1/2/3) -> Arch Complexity display text -
# used wherever a chart needs to label a bucket keyed by that raw value (not a 0-indexed
# model class index, which DISPLAY_LABELS above covers).
COMPLEXITY_CLASS_LABELS = {1: "Class I", 2: "Class II", 3: "Class III"}

# Fixed display order/labels for every metric key compute_metrics() can produce - charts only
# show whichever of these are actually present in a given metrics dict. F1 leads as the
# recommended headline metric (class-imbalance-robust, unlike accuracy).
METRIC_DISPLAY = [("f1", "F1"), ("accuracy", "Accuracy"), ("precision", "Precision"),
                   ("recall", "Recall")]

# dataviz reference palette - matches functions/graph/tooth.py
BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEAL = "#1f9d8a"
PURPLE = "#8858c8"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
METRIC_COLORS = {"accuracy": BLUE, "precision": ORANGE, "recall": TEAL, "f1": PURPLE}
SERIES_COLORS = {"lower": BLUE, "upper": TEAL, "pooled": ORANGE}

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


def annotate_bars(ax, bars, fontsize=9):
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=fontsize, color=INK)


def evaluate_jaw(jaw, ArchComplexityTransformer, ArchComplexityDataset):
    test_cache = CACHE_DIR / f"{jaw}_test.pt"
    weights_path = MODEL_DIR / f"{jaw}_best.pth"
    if not test_cache.exists() or not weights_path.exists():
        print(f"Skipping {jaw}: missing cache ({test_cache}) or weights ({weights_path}). "
              f"Run ViT/complexity/build_dataset.py and train.py first.")
        return None

    dataset = ArchComplexityDataset(test_cache)
    if len(dataset) == 0:
        print(f"Skipping {jaw}: empty test cache.")
        return None

    model = ArchComplexityTransformer(num_classes=len(DISPLAY_LABELS))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            geom, complexity = dataset[i]
            geom = geom.unsqueeze(0).to(device)
            key_padding_mask = torch.zeros(1, geom.size(1), dtype=torch.bool, device=device)
            logits = model(geom, key_padding_mask)
            probs = torch.softmax(logits, dim=1)[0]
            y_pred.append(int(torch.argmax(probs).item()))
            y_true.append(int(complexity.item()))

    return {"y_true": y_true, "y_pred": y_pred, "n": len(y_true)}


def compute_metrics(y_true, y_pred, labels=None):
    """F1 (macro, primary)/Accuracy/Precision/Recall for a hard-label prediction. labels pins the
    macro average to a fixed class set - needed wherever y_pred can contain tooth.py's
    WRONG_QUADRANT sentinel (6), which never appears in y_true, so leaving labels unset would let
    sklearn silently add a phantom all-zero class to the average and understate the score."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }


def plot_metric_bars(metrics, title, out_path):
    present = [(key, label) for key, label in METRIC_DISPLAY if key in metrics]
    labels = [label for _, label in present]
    values = [metrics[key] for key, _ in present]
    colors = [METRIC_COLORS[key] for key, _ in present]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    bars = ax.bar(labels, values, color=colors, width=0.5, zorder=3)
    annotate_bars(ax, bars)

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", color=INK)
    ax.set_title(title, color=INK)
    style_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close()


def plot_confusion(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(DISPLAY_LABELS))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DISPLAY_LABELS)
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=SURFACE)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    ax.set_title(title, color=INK)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close()


def plot_metrics_by_jaw(per_jaw_metrics, title, out_path):
    """Grouped bar chart: one group per jaw (+ pooled), one bar per metric within each group -
    shows every metric present (Accuracy/Precision/Recall/F1/mAP) per jaw side by side, not
    just accuracy. Degrades gracefully to a single accuracy bar per jaw when that's the only
    metric supplied."""
    jaws = list(per_jaw_metrics.keys())
    present = [(key, label) for key, label in METRIC_DISPLAY if all(key in per_jaw_metrics[j] for j in jaws)]

    x = list(range(len(jaws)))
    n_series = len(present)
    bar_width = 0.8 / max(n_series, 1)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    for i, (key, label) in enumerate(present):
        offsets = [xi + (i - (n_series - 1) / 2) * bar_width for xi in x]
        values = [per_jaw_metrics[j][key] for j in jaws]
        bars = ax.bar(offsets, values, width=bar_width, color=METRIC_COLORS[key], label=label, zorder=3)
        annotate_bars(ax, bars, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([j.capitalize() for j in jaws])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", color=INK)
    ax.set_title(title, color=INK)
    ax.legend(frameon=False, labelcolor=INK)
    style_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close()


def plot_single_metric_by_group(results, title, ylabel, out_path, tick_label_fn=str, xlabel="Group"):
    """Grouped bar chart: one group per bucket key (Arch Complexity, tooth count, ...),
    one bar per jaw + pooled within each group. Used for metrics that only make sense as a
    single number - e.g. arch-level exact-match accuracy has no natural precision/recall
    framing (it's a boolean per-arch flag, not a per-class prediction)."""
    series_order = [j for j in ("lower", "upper", "pooled") if results.get(j)]
    keys = sorted({k for j in series_order for k in results[j].keys()})

    x = list(range(len(keys)))
    n_series = len(series_order)
    bar_width = 0.8 / max(n_series, 1)

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.2), 5), facecolor=SURFACE)
    for i, series in enumerate(series_order):
        offsets = [xi + (i - (n_series - 1) / 2) * bar_width for xi in x]
        values = [results[series].get(k, 0.0) for k in keys]
        bars = ax.bar(offsets, values, width=bar_width, color=SERIES_COLORS[series],
                       label=series.capitalize(), zorder=3)
        annotate_bars(ax, bars, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([tick_label_fn(k) for k in keys])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, color=INK)
    ax.legend(frameon=False, labelcolor=INK)
    style_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close()


def plot_metrics_by_group(results, title, out_path, tick_label_fn=str, xlabel="Group"):
    """Small multiples: one subplot per metric (Accuracy/Precision/Recall/F1), each a grouped
    bar chart with one group per bucket key and one bar per jaw + pooled - mirrors
    tooth.py's plot_metrics_by_number, generalized to an arbitrary jaw/pooled series set.
    results: {jaw/"pooled": {bucket_key: {metric: value}}}."""
    series_order = [j for j in ("lower", "upper", "pooled") if results.get(j)]
    keys = sorted({k for j in series_order for k in results[j].keys()})
    metric_names = [key for key, _ in METRIC_DISPLAY
                    if any(key in results[j].get(k, {}) for j in series_order for k in keys)]

    x = list(range(len(keys)))
    n_series = len(series_order)
    bar_width = 0.8 / max(n_series, 1)

    fig, axes = plt.subplots(len(metric_names), 1,
                              figsize=(max(10, len(keys) * 1.4), 4.5 * len(metric_names)),
                              facecolor=SURFACE)
    if len(metric_names) == 1:
        axes = [axes]

    for ax, metric_key in zip(axes, metric_names):
        metric_label = dict(METRIC_DISPLAY)[metric_key]
        for i, series in enumerate(series_order):
            offsets = [xi + (i - (n_series - 1) / 2) * bar_width for xi in x]
            values = [results[series].get(k, {}).get(metric_key, 0.0) for k in keys]
            bars = ax.bar(offsets, values, width=bar_width, color=SERIES_COLORS[series],
                           label=series.capitalize(), zorder=3)
            annotate_bars(ax, bars, fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([tick_label_fn(k) for k in keys])
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(metric_label, color=INK)
        style_axes(ax)
        ax.legend(frameon=False, labelcolor=INK, fontsize=8)

    axes[-1].set_xlabel(xlabel, color=INK)
    fig.suptitle(title, color=INK, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close()


def _bucketed_metrics(items, labels=None):
    """items: iterable of (true, pred, bucket_key). Returns {bucket_key: metrics} using
    compute_metrics (no probability scores available this deep in the pipeline - see
    compute_metrics docstring)."""
    buckets = defaultdict(lambda: {"true": [], "pred": []})
    for true, pred, key in items:
        buckets[key]["true"].append(true)
        buckets[key]["pred"].append(pred)
    return {k: compute_metrics(b["true"], b["pred"], labels=labels) for k, b in sorted(buckets.items())}


# Every function below scores against tooth.gate_by_tens(y_postproc, y_tens_pred_hungarian,
# y_fdi_number) rather than raw y_postproc - a tooth whose predicted FDI *tens* digit was wrong
# gets replaced with the WRONG_QUADRANT sentinel (6) first, same as every chart in tooth.py. Before
# this fix these three functions compared y_true directly against y_postproc (last-digit only),
# so a wrong-quadrant tooth whose last digit happened to still be right was silently scored as a
# hit here even though tooth.py's own charts (and production) would show it as the wrong FDI
# number entirely - the complexity/tooth-count breakdowns below now agree with tooth.py instead of
# quietly assuming GT tens digit like every chart did before this session's tens-gating fix.


def evaluate_fdi_accuracy_by_complexity(per_jaw, complexity_by_image, gate_by_tens):
    """Groups the ResNet+Hungarian pipeline's per-tooth FDI-number performance (test set) by
    each arch's ground-truth Arch Complexity, so Accuracy/Precision/Recall/F1 can be read as
    a function of case difficulty rather than just pooled across the whole test set. Takes the
    already-computed tooth.evaluate_tooth_number() result and arch->complexity lookup (see
    main()) instead of recomputing either - both are expensive (a full ResNet forward pass
    over the test set) and shared with evaluate_arch_accuracy. gate_by_tens is tooth.gate_by_tens,
    passed in rather than imported so this module doesn't need its own copy."""
    per_jaw_by_complexity = {}
    all_items = []

    for jaw in ("lower", "upper"):
        d = per_jaw[jaw]
        gated_pred = gate_by_tens(d["y_postproc"], d["y_tens_pred_hungarian"], d["y_fdi_number"])
        items = []
        for true_digit, pred_digit, image_name in zip(d["y_true"], gated_pred, d["y_image_name"]):
            complexity = complexity_by_image.get(image_name)
            if complexity is None:
                continue
            items.append((true_digit, pred_digit, complexity))
        per_jaw_by_complexity[jaw] = _bucketed_metrics(items, labels=list(range(6)))
        all_items.extend(items)

    per_jaw_by_complexity["pooled"] = _bucketed_metrics(all_items, labels=list(range(6)))
    return per_jaw_by_complexity


def evaluate_fdi_accuracy_by_tooth_count(per_jaw, gate_by_tens):
    """Same idea as evaluate_fdi_accuracy_by_complexity, but bucketed by how many teeth were
    detected in the arch a tooth came from - fewer detected teeth means a smaller quadrant group
    for the Hungarian duplicate-resolution post-process (ResNet/tooth/postprocess.py) to work
    with, so performance is expected to shift as tooth count drops."""
    per_jaw_by_count = {}
    all_items = []

    for jaw in ("lower", "upper"):
        d = per_jaw[jaw]
        gated_pred = gate_by_tens(d["y_postproc"], d["y_tens_pred_hungarian"], d["y_fdi_number"])
        tooth_count = defaultdict(int)
        for image_name in d["y_image_name"]:
            tooth_count[image_name] += 1

        items = [
            (true_digit, pred_digit, tooth_count[image_name])
            for true_digit, pred_digit, image_name in zip(d["y_true"], gated_pred, d["y_image_name"])
        ]
        per_jaw_by_count[jaw] = _bucketed_metrics(items, labels=list(range(6)))
        all_items.extend(items)

    per_jaw_by_count["pooled"] = _bucketed_metrics(all_items, labels=list(range(6)))
    return per_jaw_by_count


def evaluate_arch_accuracy(per_jaw, complexity_by_image, gate_by_tens):
    """For each arch (oral image) in the test set, checks whether EVERY detected tooth's full FDI
    number - tens digit included, via gate_by_tens - was predicted correctly - a stricter metric
    than per-tooth accuracy, since one wrong tooth invalidates the whole arch. This is a boolean
    per-arch flag, not a per-class prediction, so only accuracy is meaningful here (no natural
    precision/recall/F1 framing). Takes the already-computed tooth.evaluate_tooth_number() result
    and arch->complexity lookup (see main()) instead of recomputing either.

    Returns (per_jaw_accuracy, per_jaw_by_complexity, per_jaw_by_tooth_count):
      per_jaw_accuracy: {jaw/"pooled": {"accuracy": float, "n": int}} - n is arch count.
      per_jaw_by_complexity: {jaw/"pooled": {complexity: accuracy}}.
      per_jaw_by_tooth_count: {jaw/"pooled": {tooth_count: accuracy}}.
    """
    # One record per arch, tagged with its jaw - everything below (per-jaw, pooled, by
    # complexity, by tooth count) is just a filter/group-by over this single flat list.
    records = []
    for jaw in ("lower", "upper"):
        d = per_jaw[jaw]
        gated_pred = gate_by_tens(d["y_postproc"], d["y_tens_pred_hungarian"], d["y_fdi_number"])
        arches = defaultdict(lambda: {"correct": True, "count": 0, "complexity": None})
        for true_digit, pred_digit, image_name in zip(d["y_true"], gated_pred, d["y_image_name"]):
            arch = arches[image_name]
            if true_digit != pred_digit:
                arch["correct"] = False
            arch["count"] += 1
            if arch["complexity"] is None:
                arch["complexity"] = complexity_by_image.get(image_name)
        for a in arches.values():
            records.append({"jaw": jaw, **a})

    def accuracy_of(recs):
        return sum(r["correct"] for r in recs) / len(recs) if recs else 0.0

    per_jaw_accuracy = {}
    per_jaw_by_complexity = {}
    per_jaw_by_tooth_count = {}

    for series in ("lower", "upper", "pooled"):
        recs = records if series == "pooled" else [r for r in records if r["jaw"] == series]
        per_jaw_accuracy[series] = {"accuracy": accuracy_of(recs), "n": len(recs)}

        by_complexity = defaultdict(list)
        for r in recs:
            if r["complexity"] is not None:
                by_complexity[r["complexity"]].append(r)
        per_jaw_by_complexity[series] = {c: accuracy_of(rs) for c, rs in sorted(by_complexity.items())}

        by_count = defaultdict(list)
        for r in recs:
            by_count[r["count"]].append(r)
        per_jaw_by_tooth_count[series] = {c: accuracy_of(rs) for c, rs in sorted(by_count.items())}

    return per_jaw_accuracy, per_jaw_by_complexity, per_jaw_by_tooth_count


def main():
    model_module = load_module("vit_complexity_model_eval", COMPLEXITY_DIR / "model.py")
    dataset_module = load_module("vit_complexity_dataset_eval", COMPLEXITY_DIR / "dataset.py")
    ArchComplexityTransformer = model_module.ArchComplexityTransformer
    ArchComplexityDataset = dataset_module.ArchComplexityDataset

    summary = {}
    per_jaw_metrics = {}
    all_true, all_pred = [], []

    for jaw in ("lower", "upper"):
        print(f"\nEvaluating {jaw} Arch Complexity model on test split...")
        result = evaluate_jaw(jaw, ArchComplexityTransformer, ArchComplexityDataset)
        if result is None:
            continue

        metrics = compute_metrics(result["y_true"], result["y_pred"])
        per_jaw_metrics[jaw] = metrics
        summary[jaw] = {**metrics, "n": result["n"]}
        print(f"{jaw}: n={result['n']}, f1={metrics['f1']:.4f}, accuracy={metrics['accuracy']:.4f}, "
              f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}")

        plot_metric_bars(
            metrics,
            f"Arch Complexity Performance - {jaw.capitalize()} Jaw (Test Set)",
            RESULTS_DIR / f"complexity_metrics_{jaw}.png"
        )
        plot_confusion(
            result["y_true"], result["y_pred"],
            f"Arch Complexity Confusion Matrix - {jaw.capitalize()} Jaw (Test Set)",
            RESULTS_DIR / f"complexity_confusion_{jaw}.png"
        )

        all_true.extend(result["y_true"])
        all_pred.extend(result["y_pred"])

    if all_true:
        pooled_metrics = compute_metrics(all_true, all_pred)
        per_jaw_metrics["pooled"] = pooled_metrics
        summary["pooled"] = {**pooled_metrics, "n": len(all_true)}
        print(f"\nPooled (both jaws): n={len(all_true)}, f1={pooled_metrics['f1']:.4f}, "
              f"accuracy={pooled_metrics['accuracy']:.4f}")

        plot_metric_bars(
            pooled_metrics,
            "Arch Complexity Performance - Pooled (Test Set)",
            RESULTS_DIR / "complexity_metrics_pooled.png"
        )
        plot_confusion(
            all_true, all_pred,
            "Arch Complexity Confusion Matrix - Pooled (Test Set)",
            RESULTS_DIR / "complexity_confusion_pooled.png"
        )

    if per_jaw_metrics:
        plot_metrics_by_jaw(
            per_jaw_metrics,
            "Arch Complexity Performance by Jaw (Test Set)",
            RESULTS_DIR / "complexity_accuracy_by_jaw.png"
        )

    # The FDI accuracy/arch-accuracy breakdowns below all need the full ResNet+Hungarian
    # tooth-number pipeline result and the arch->complexity lookup - computed once here and
    # shared, since evaluate_tooth_number() re-runs a full ResNet forward pass over every test
    # image and shouldn't be paid for more than once.
    eval_module = load_module("tooth_eval", GRAPH_DIR / "tooth.py")
    complexity_build = load_module("vit_complexity_build_eval", COMPLEXITY_DIR / "build_dataset.py")
    dataset_dir = PROJECT_ROOT.parent / "dataset"

    print("\n" + "=" * 70)
    print("Running ResNet+Hungarian tooth-number pipeline on the test set...")
    print("=" * 70)
    per_jaw_tooth_number = eval_module.evaluate_tooth_number()
    complexity_by_image = complexity_build.load_complexity_labels(dataset_dir, "test")

    print("\n" + "=" * 70)
    print("FDI tooth-number performance by Arch Complexity (ResNet+Hungarian, Test Set)")
    print("=" * 70)
    fdi_by_complexity = evaluate_fdi_accuracy_by_complexity(
        per_jaw_tooth_number, complexity_by_image, eval_module.gate_by_tens
    )
    for series, buckets in fdi_by_complexity.items():
        breakdown = ", ".join(f"{COMPLEXITY_CLASS_LABELS.get(c, c)}: f1={m['f1']:.4f} acc={m['accuracy']:.4f}"
                               for c, m in sorted(buckets.items()))
        print(f"{series}: {breakdown}")
    plot_metrics_by_group(
        fdi_by_complexity,
        "FDI Tooth-Number Performance by Arch Complexity (Test Set)",
        RESULTS_DIR / "complexity_fdi_accuracy_by_complexity.png",
        tick_label_fn=lambda c: COMPLEXITY_CLASS_LABELS.get(c, str(c)),
        xlabel="Arch Complexity",
    )
    summary["fdi_accuracy_by_complexity"] = fdi_by_complexity

    print("\n" + "=" * 70)
    print("FDI tooth-number performance by detected tooth count (ResNet+Hungarian, Test Set)")
    print("=" * 70)
    fdi_by_tooth_count = evaluate_fdi_accuracy_by_tooth_count(per_jaw_tooth_number, eval_module.gate_by_tens)
    for series, buckets in fdi_by_tooth_count.items():
        breakdown = ", ".join(f"{c} teeth: f1={m['f1']:.4f} acc={m['accuracy']:.4f}"
                               for c, m in sorted(buckets.items()))
        print(f"{series}: {breakdown}")
    plot_metrics_by_group(
        fdi_by_tooth_count,
        "FDI Tooth-Number Performance by Detected Tooth Count (Test Set)",
        RESULTS_DIR / "complexity_fdi_accuracy_by_tooth_count.png",
        xlabel="Detected Tooth Count",
    )
    summary["fdi_accuracy_by_tooth_count"] = fdi_by_tooth_count

    print("\n" + "=" * 70)
    print("Arch-level accuracy (every tooth in the arch correct) - Test Set")
    print("=" * 70)
    arch_accuracy, arch_accuracy_by_complexity, arch_accuracy_by_tooth_count = evaluate_arch_accuracy(
        per_jaw_tooth_number, complexity_by_image, eval_module.gate_by_tens
    )
    for jaw, m in arch_accuracy.items():
        print(f"{jaw}: n={m['n']}, accuracy={m['accuracy']:.4f}")
    for series, accs in arch_accuracy_by_complexity.items():
        breakdown = ", ".join(f"{COMPLEXITY_CLASS_LABELS.get(c, c)}: {a:.4f}" for c, a in sorted(accs.items()))
        print(f"{series}: {breakdown}")
    for series, accs in arch_accuracy_by_tooth_count.items():
        breakdown = ", ".join(f"{c} teeth: {a:.4f}" for c, a in sorted(accs.items()))
        print(f"{series}: {breakdown}")

    plot_metrics_by_jaw(
        {j: {"accuracy": m["accuracy"]} for j, m in arch_accuracy.items()},
        "Arch-Level Accuracy - Every Tooth Correct (Test Set)",
        RESULTS_DIR / "complexity_arch_accuracy_by_jaw.png"
    )
    plot_single_metric_by_group(
        arch_accuracy_by_complexity,
        "Arch-Level Accuracy by Arch Complexity (Test Set)",
        "Arch-Level Accuracy (All Teeth Correct)",
        RESULTS_DIR / "complexity_arch_accuracy_by_complexity.png",
        tick_label_fn=lambda c: COMPLEXITY_CLASS_LABELS.get(c, str(c)),
        xlabel="Arch Complexity",
    )
    plot_single_metric_by_group(
        arch_accuracy_by_tooth_count,
        "Arch-Level Accuracy by Detected Tooth Count (Test Set)",
        "Arch-Level Accuracy (All Teeth Correct)",
        RESULTS_DIR / "complexity_arch_accuracy_by_tooth_count.png",
        xlabel="Detected Tooth Count",
    )
    summary["arch_accuracy"] = arch_accuracy
    summary["arch_accuracy_by_complexity"] = arch_accuracy_by_complexity
    summary["arch_accuracy_by_tooth_count"] = arch_accuracy_by_tooth_count

    with open(RESULTS_DIR / "complexity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone. Results saved to", RESULTS_DIR.resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
