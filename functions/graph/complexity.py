"""
Evaluates the ViT/complexity arch-level complexity classifier (class I/II/III, predicted from
per-tooth geometry alone - see ViT/complexity/model.py) on the held-out test split
(dataset/test/, never touched by training or validation).

Reuses ViT/complexity/{model,dataset}.py and the {jaw}_test.pt caches already built by
ViT/complexity/build_dataset.py instead of reimplementing the data pipeline.

Usage:
    .venv/Scripts/python.exe functions/graph/complexity.py
"""
import json
import importlib.util
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

DISPLAY_LABELS = ["1", "2", "3"]  # matches ViT/complexity/train.py's DISPLAY_LABELS

# dataviz reference palette - matches functions/graph/evaluate_test_performance.py
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
            y_pred.append(int(torch.argmax(logits, dim=1).item()))
            y_true.append(int(complexity.item()))

    return {"y_true": y_true, "y_pred": y_pred, "n": len(y_true)}


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def plot_metric_bars(metrics, title, out_path):
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    values = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]]

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


def plot_confusion(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(DISPLAY_LABELS))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DISPLAY_LABELS)
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=SURFACE)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    ax.set_title(title, color=INK)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close()


def plot_accuracy_by_jaw(per_jaw_metrics, title, out_path):
    """Grouped bar chart of accuracy per jaw + pooled, so lower vs upper performance is visible
    at a glance - pooled gets the accent color to set it apart from the two per-jaw bars."""
    jaws = list(per_jaw_metrics.keys())
    accs = [per_jaw_metrics[j]["accuracy"] for j in jaws]
    colors = [ORANGE if j == "pooled" else BLUE for j in jaws]

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=SURFACE)
    bars = ax.bar([j.capitalize() for j in jaws], accs, color=colors, width=0.5, zorder=3)
    annotate_bars(ax, bars)

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy", color=INK)
    ax.set_title(title, color=INK)
    style_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close()


def main():
    model_module = load_module("vit_complexity_model_eval", COMPLEXITY_DIR / "model.py")
    dataset_module = load_module("vit_complexity_dataset_eval", COMPLEXITY_DIR / "dataset.py")
    ArchComplexityTransformer = model_module.ArchComplexityTransformer
    ArchComplexityDataset = dataset_module.ArchComplexityDataset

    summary = {}
    per_jaw_metrics = {}
    all_true, all_pred = [], []

    for jaw in ("lower", "upper"):
        print(f"\nEvaluating {jaw} complexity classifier on test split...")
        result = evaluate_jaw(jaw, ArchComplexityTransformer, ArchComplexityDataset)
        if result is None:
            continue

        metrics = compute_metrics(result["y_true"], result["y_pred"])
        per_jaw_metrics[jaw] = metrics
        summary[jaw] = {**metrics, "n": result["n"]}
        print(f"{jaw}: n={result['n']}, accuracy={metrics['accuracy']:.4f}, "
              f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
              f"f1={metrics['f1']:.4f}")

        plot_metric_bars(
            metrics,
            f"Complexity Classification Performance - {jaw.capitalize()} Jaw (Test Set)",
            RESULTS_DIR / f"complexity_metrics_{jaw}.png"
        )
        plot_confusion(
            result["y_true"], result["y_pred"],
            f"Complexity Confusion Matrix - {jaw.capitalize()} Jaw (Test Set)",
            RESULTS_DIR / f"complexity_confusion_{jaw}.png"
        )

        all_true.extend(result["y_true"])
        all_pred.extend(result["y_pred"])

    if all_true:
        pooled_metrics = compute_metrics(all_true, all_pred)
        per_jaw_metrics["pooled"] = pooled_metrics
        summary["pooled"] = {**pooled_metrics, "n": len(all_true)}
        print(f"\nPooled (both jaws): n={len(all_true)}, accuracy={pooled_metrics['accuracy']:.4f}")

        plot_metric_bars(
            pooled_metrics,
            "Complexity Classification Performance - Pooled (Test Set)",
            RESULTS_DIR / "complexity_metrics_pooled.png"
        )
        plot_confusion(
            all_true, all_pred,
            "Complexity Confusion Matrix - Pooled (Test Set)",
            RESULTS_DIR / "complexity_confusion_pooled.png"
        )

    if per_jaw_metrics:
        plot_accuracy_by_jaw(
            per_jaw_metrics,
            "Complexity Accuracy by Jaw (Test Set)",
            RESULTS_DIR / "complexity_accuracy_by_jaw.png"
        )

    with open(RESULTS_DIR / "complexity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone. Results saved to", RESULTS_DIR.resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
