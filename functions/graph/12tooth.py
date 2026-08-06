"""
Compares four ways of assigning each tooth its FDI number, all measured against the same
per-tooth accuracy and macro F1 (predicted FDI number == true FDI number, test set):

  1. X-Sort            - sort centroids by x-coordinate, ascending. No notion of "path" at all.
  2. Nearest Neighbor   - greedy: start at the leftmost tooth, repeatedly hop to whichever
                           unvisited tooth is closest.
  3. Held-Karp          - exact O(2^n * n^2) dynamic programming for the shortest Hamiltonian
                           path starting at the leftmost tooth - what js/geometry.js's
                           computeArchOrder actually uses in production.
  4. ResNet+ViT         - the trained pipeline itself (tooth.py's evaluate_tooth_number): a
                           ResNet classifies each detected tooth from its image crop, then a ViT
                           arch transformer refines all of an arch's predictions jointly. Unlike
                           the geometry-only baselines above, this one actually looks at the
                           tooth images - it doesn't need an explicit ordering step at all.

An MST-based order (Prim's MST from the leftmost tooth, then DFS preorder) was tried too, but
on this dataset it produces the exact same order as Nearest Neighbor on every arch checked
(dental arches are close enough to a simple curve that the two never diverge in practice), so
it added a redundant bar with no new information and was dropped from the comparison.

The first three only ever see raw centroid (x, y) positions - no PCA arch alignment, no trained
model, no gap-aware slot reconstruction (computeArchSlots) - then read the resulting order off
the canonical 12-slot FDI list ([46..41,31..36] / [16..11,21..26], the same left-to-right order
js/render.js's drawArchPath and drawFdiNumbers use). This isolates how much of the production
pipeline's accuracy comes from ordering algorithm choice alone, holding everything else fixed.
ResNet+ViT accuracy, by contrast, is the last-digit accuracy tooth.py already computes (tens
digit taken from ground truth in both cases, so "last digit correct" == "full FDI number
correct" here) - it's the actual accuracy the app gets today, for comparison against what pure
geometry could achieve.

Ground truth source: dataset/test/labels_json/{jaw}/*.json (same GT the rest of
functions/graph uses) for the three geometry baselines - reads each tooth's `teeth_num` and
`segmentation` polygon directly, no other pipeline code involved. ResNet+ViT reuses tooth.py's
evaluate_tooth_number(), which pulls from dataset/test/images/{jaw} and the cached feature CSVs
under ResNet/tooth/csv/ (same test split, no data leakage between the two).

Usage:
    .venv/Scripts/python.exe functions/graph/12tooth.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

GRAPH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GRAPH_DIR.parent.parent          # .../CHAI/CHAI
DATASET_DIR = PROJECT_ROOT.parent / "dataset"   # .../CHAI/dataset
RESULTS_DIR = GRAPH_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Canonical left-to-right (ascending x) slot order, 12 per arch - mirrors js/render.js's
# fdiLabels arrays exactly, which assume the standard convention of the patient's right side
# appearing on the left of the image. Only digits 1-6 (incisor through first molar) are
# covered; wisdom teeth (7/8) aren't part of this fixed slot scheme.
CANONICAL_ORDER = {
    "lower": [46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36],
    "upper": [16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26],
}

# The pure-geometry ordering algorithms (evaluate_jaw loops over these). ResNet+ViT is
# evaluated separately (it isn't an ordering algorithm) and merged in for the comparison plot.
GEOMETRY_ALGORITHMS = ["x_sort", "nearest_neighbor", "held_karp"]
ALL_ALGORITHMS = GEOMETRY_ALGORITHMS + ["resnet_vit"]
ALGORITHM_LABELS = {
    "x_sort": "X-Sort",
    "nearest_neighbor": "Nearest Neighbor",
    "held_karp": "Held-Karp",
    "resnet_vit": "ResNet+ViT",
}

# dataviz reference palette - matches functions/graph/tooth.py and complexity.py
BLUE = "#2a78d6"
ORANGE = "#eb6834"
PURPLE = "#8858c8"
PINK = "#d6538a"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
ALGORITHM_COLORS = {
    "x_sort": BLUE, "nearest_neighbor": ORANGE, "held_karp": PURPLE, "resnet_vit": PINK,
}


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


def compute_f1(y_true_digit, y_pred_digit):
    """Macro F1 over whatever digit classes actually appear in y_true - unlike accuracy, F1
    penalizes a method that does well on common classes (e.g. incisors) but poorly on rare ones
    (e.g. missing/rare tooth positions) equally across classes rather than by frequency."""
    if not y_true_digit:
        return 0.0
    labels = sorted(set(y_true_digit) | set(y_pred_digit))
    return float(f1_score(y_true_digit, y_pred_digit, labels=labels, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------
def load_arch_teeth(json_path):
    """Returns [(teeth_num, x, y), ...] for teeth whose FDI last digit is 1-6, read straight
    from a labels_json file - same GT source and format as tooth.py's
    _load_gt_boxes_with_numbers, just reduced to a centroid instead of a bbox."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    teeth = []
    for t in data.get("tooth", []):
        num = t.get("teeth_num")
        seg = t.get("segmentation", [])
        if num is None or not seg or not (1 <= num % 10 <= 6):
            continue
        poly = np.array(seg, dtype=np.float64).reshape(-1, 2)
        cx, cy = poly.mean(axis=0)
        teeth.append((int(num), float(cx), float(cy)))
    return teeth


# ---------------------------------------------------------------------------
# Ordering algorithms - each takes a list of (x, y) points and returns a list of point
# indices, "leftmost tooth first" through to the rightmost, per its own strategy.
# ---------------------------------------------------------------------------
def x_sort_order(points):
    return sorted(range(len(points)), key=lambda i: points[i][0])


def nearest_neighbor_order(points):
    n = len(points)
    start = min(range(n), key=lambda i: points[i][0])
    visited = [False] * n
    visited[start] = True
    order = [start]
    current = start
    for _ in range(n - 1):
        cx, cy = points[current]
        best_j, best_d = -1, float('inf')
        for j in range(n):
            if visited[j]:
                continue
            dx, dy = points[j][0] - cx, points[j][1] - cy
            d = dx * dx + dy * dy
            if d < best_d:
                best_d, best_j = d, j
        visited[best_j] = True
        order.append(best_j)
        current = best_j
    return order


def held_karp_order(points):
    """Exact shortest Hamiltonian path starting at the leftmost tooth, via bitmask DP - mirrors
    js/geometry.js's computeArchOrder (same algorithm, just without the PCA-rotated start-point
    selection, since this script works in raw centroid space like the other baselines)."""
    n = len(points)
    if n <= 1:
        return list(range(n))
    start = min(range(n), key=lambda i: points[i][0])
    others = [i for i in range(n) if i != start]
    m = len(others)

    def dist(a, b):
        dx, dy = points[a][0] - points[b][0], points[a][1] - points[b][1]
        return (dx * dx + dy * dy) ** 0.5

    pair_dist = [[dist(others[a], others[b]) for b in range(m)] for a in range(m)]
    start_dist = [dist(start, others[j]) for j in range(m)]

    size = 1 << m
    INF = float('inf')
    dp = [[INF] * m for _ in range(size)]
    parent = [[-1] * m for _ in range(size)]
    for j in range(m):
        dp[1 << j][j] = start_dist[j]

    for mask in range(size):
        row = dp[mask]
        for j in range(m):
            cur = row[j]
            if cur == INF or not (mask & (1 << j)):
                continue
            dj = pair_dist[j]
            for k in range(m):
                if mask & (1 << k):
                    continue
                new_mask = mask | (1 << k)
                new_cost = cur + dj[k]
                if new_cost < dp[new_mask][k]:
                    dp[new_mask][k] = new_cost
                    parent[new_mask][k] = j

    full = size - 1
    best_j = min(range(m), key=lambda j: dp[full][j])

    path = []
    mask, j = full, best_j
    while j != -1:
        path.append(others[j])
        prev_j = parent[mask][j]
        mask ^= (1 << j)
        j = prev_j
    path.reverse()
    return [start] + path


ORDER_FUNCS = {
    "x_sort": x_sort_order,
    "nearest_neighbor": nearest_neighbor_order,
    "held_karp": held_karp_order,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_jaw(jaw):
    json_dir = DATASET_DIR / "test" / "labels_json" / jaw
    slots = CANONICAL_ORDER[jaw]

    per_algo = {algo: {"y_true": [], "y_pred": [], "y_true_digit": [], "y_pred_digit": []}
                for algo in GEOMETRY_ALGORITHMS}
    n_arches = 0

    for json_path in sorted(json_dir.glob("*.json")):
        teeth = load_arch_teeth(json_path)
        if not teeth:
            continue
        n_arches += 1
        true_nums = [t[0] for t in teeth]
        points = [(t[1], t[2]) for t in teeth]

        for algo in GEOMETRY_ALGORITHMS:
            order_idx = ORDER_FUNCS[algo](points)
            bucket = per_algo[algo]
            for i, idx in enumerate(order_idx):
                if i >= len(slots):
                    break  # more GT teeth than canonical slots - nothing left to compare against
                pred_num = slots[i]
                true_num = true_nums[idx]
                bucket["y_true"].append(true_num)
                bucket["y_pred"].append(pred_num)
                bucket["y_true_digit"].append(true_num % 10)
                bucket["y_pred_digit"].append(pred_num % 10)

    result = {"n_arches": n_arches}
    for algo in GEOMETRY_ALGORITHMS:
        b = per_algo[algo]
        n_teeth = len(b["y_true"])
        accuracy = sum(t == p for t, p in zip(b["y_true"], b["y_pred"])) / n_teeth if n_teeth else 0.0
        f1 = compute_f1(b["y_true_digit"], b["y_pred_digit"])
        result[algo] = {**b, "accuracy": accuracy, "f1": f1, "n_teeth": n_teeth}
    return result


def pool_jaws(per_jaw):
    pooled = {"n_arches": per_jaw["lower"]["n_arches"] + per_jaw["upper"]["n_arches"]}
    for algo in GEOMETRY_ALGORITHMS:
        y_true = per_jaw["lower"][algo]["y_true"] + per_jaw["upper"][algo]["y_true"]
        y_pred = per_jaw["lower"][algo]["y_pred"] + per_jaw["upper"][algo]["y_pred"]
        y_true_digit = per_jaw["lower"][algo]["y_true_digit"] + per_jaw["upper"][algo]["y_true_digit"]
        y_pred_digit = per_jaw["lower"][algo]["y_pred_digit"] + per_jaw["upper"][algo]["y_pred_digit"]
        n_teeth = len(y_true)
        accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / n_teeth if n_teeth else 0.0
        f1 = compute_f1(y_true_digit, y_pred_digit)
        pooled[algo] = {"y_true": y_true, "y_pred": y_pred,
                         "y_true_digit": y_true_digit, "y_pred_digit": y_pred_digit,
                         "accuracy": accuracy, "f1": f1, "n_teeth": n_teeth}
    return pooled


def evaluate_resnet_vit():
    """Runs the trained pipeline itself (tooth.py's evaluate_tooth_number: ResNet per-tooth
    classification + ViT arch-transformer refinement) on the test set and reduces it to the
    same per-tooth accuracy/F1 the geometry baselines report. Reuses tooth.py rather than
    reimplementing it, same pattern as complexity.py's main()."""
    tooth_eval = load_module("tooth_eval_for_12tooth", GRAPH_DIR / "tooth.py")
    per_jaw_raw = tooth_eval.evaluate_tooth_number()

    result = {}
    all_true, all_pred = [], []
    for jaw in ("lower", "upper"):
        y_true = per_jaw_raw[jaw]["y_true"]
        y_pred = per_jaw_raw[jaw]["y_refined"]
        n_teeth = len(y_true)
        accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / n_teeth if n_teeth else 0.0
        f1 = compute_f1(y_true, y_pred)
        result[jaw] = {"y_true_digit": y_true, "y_pred_digit": y_pred,
                        "accuracy": accuracy, "f1": f1, "n_teeth": n_teeth}
        all_true += y_true
        all_pred += y_pred

    n_pooled = len(all_true)
    pooled_accuracy = sum(t == p for t, p in zip(all_true, all_pred)) / n_pooled if n_pooled else 0.0
    pooled_f1 = compute_f1(all_true, all_pred)
    result["pooled"] = {"y_true_digit": all_true, "y_pred_digit": all_pred,
                         "accuracy": pooled_accuracy, "f1": pooled_f1, "n_teeth": n_pooled}
    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_metrics_comparison(per_jaw, out_path):
    """Two stacked grouped-bar panels in one figure - Accuracy on top, macro F1 on the bottom -
    so both metrics can be read side by side without switching between image files. Each panel
    groups by jaw (+ pooled), with one bar per method within each group."""
    jaws = ["lower", "upper", "pooled"]
    x = list(range(len(jaws)))
    n_series = len(ALL_ALGORITHMS)
    bar_width = 0.8 / n_series
    metrics = [("accuracy", "Accuracy"), ("f1", "F1 Score (Macro)")]

    fig, axes = plt.subplots(2, 1, figsize=(11, 11), facecolor=SURFACE)
    for ax, (metric_key, metric_label) in zip(axes, metrics):
        for i, algo in enumerate(ALL_ALGORITHMS):
            offsets = [xi + (i - (n_series - 1) / 2) * bar_width for xi in x]
            values = [per_jaw[j][algo][metric_key] for j in jaws]
            bars = ax.bar(offsets, values, width=bar_width, color=ALGORITHM_COLORS[algo],
                           label=ALGORITHM_LABELS[algo], zorder=3)
            annotate_bars(ax, bars, fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([j.capitalize() for j in jaws])
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(metric_label, color=INK)
        ax.legend(frameon=False, labelcolor=INK)
        style_axes(ax)

    fig.suptitle("Tooth Numbering Accuracy & F1 Score by Method (Test Set)", color=INK, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close()


def plot_confusion(y_true_digit, y_pred_digit, title, out_path):
    labels = list(range(1, 7))
    display_labels = [str(d) for d in labels]
    cm = confusion_matrix(y_true_digit, y_pred_digit, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=SURFACE)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    ax.set_title(title, color=INK)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close()


def main():
    per_jaw = {}
    for jaw in ("lower", "upper"):
        print(f"Evaluating {jaw} test arches (x_sort / nearest_neighbor / held_karp)...")
        result = evaluate_jaw(jaw)
        per_jaw[jaw] = result
        print(f"{jaw}: {result['n_arches']} arches - " + ", ".join(
            f"{ALGORITHM_LABELS[a]}=acc:{result[a]['accuracy']:.4f}/f1:{result[a]['f1']:.4f}"
            for a in GEOMETRY_ALGORITHMS
        ))

    per_jaw["pooled"] = pool_jaws(per_jaw)
    pooled = per_jaw["pooled"]
    print(f"pooled: {pooled['n_arches']} arches - " + ", ".join(
        f"{ALGORITHM_LABELS[a]}=acc:{pooled[a]['accuracy']:.4f}/f1:{pooled[a]['f1']:.4f}"
        for a in GEOMETRY_ALGORITHMS
    ))

    print("\nEvaluating ResNet+ViT pipeline on the test set (tooth.py's evaluate_tooth_number)...")
    resnet_vit = evaluate_resnet_vit()
    for jaw in ("lower", "upper", "pooled"):
        per_jaw[jaw]["resnet_vit"] = resnet_vit[jaw]
        print(f"{jaw}: ResNet+ViT accuracy={resnet_vit[jaw]['accuracy']:.4f} "
              f"f1={resnet_vit[jaw]['f1']:.4f} (n={resnet_vit[jaw]['n_teeth']})")

    plot_metrics_comparison(per_jaw, RESULTS_DIR / "12tooth_metrics.png")

    for jaw in ("lower", "upper"):
        d = per_jaw[jaw]["held_karp"]
        plot_confusion(d["y_true_digit"], d["y_pred_digit"],
                       f"Held-Karp Tooth Numbering Confusion - {jaw.capitalize()} Jaw (Test Set)",
                       RESULTS_DIR / f"12tooth_confusion_{jaw}.png")

    summary = {
        jaw: {
            "n_arches": per_jaw[jaw]["n_arches"],
            **{algo: {"accuracy": per_jaw[jaw][algo]["accuracy"], "f1": per_jaw[jaw][algo]["f1"],
                      "n_teeth": per_jaw[jaw][algo]["n_teeth"]}
               for algo in ALL_ALGORITHMS},
        }
        for jaw in ("lower", "upper", "pooled")
    }
    with open(RESULTS_DIR / "12tooth_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone. Results saved to", RESULTS_DIR.resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
