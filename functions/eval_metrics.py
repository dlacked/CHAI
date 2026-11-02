"""
Shared metric/bucketing helper functions reused by every model's evaluation script
(ResNet/evaluate_chai*.py, comparison/ghorbani/evaluate*.py, comparison/yoon/evaluate*.py) so all
three models are scored identically: missing-tooth-count buckets, malocclusion-complexity
buckets, tooth-level metrics by bucket, and arch-level metrics (missing-teeth / missing-vs-complete
/ complexity-by-arch). Each caller loads this module by path via its own local load_module()
helper (dynamic import, so this file doesn't need to be on sys.path or part of a package).

This module has no runnable entry point of its own (no __main__) - it is purely a library of
functions other scripts dynamically import. Renamed from functions/graph/tooth.py and moved out
of functions/graph/ on 2026-10-30 (that directory no longer exists - this module hasn't done any
plotting since the Mirrored-Variant evaluation pipeline that used to live here
(build_mirrored_arch_sequences, evaluate_tooth_number, main(), and their supporting plot helpers,
plus the now-unused load_module() this file no longer needs either) was moved out as dead code -
see backups/superseded_20260831/functions_graph_tooth_mirrored_eval.py if it's ever needed again.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# Sentinel "digit" for a tooth whose predicted FDI *tens* digit was wrong (see
# compute_tens_predictions / gate_by_tens) - distinct from any real last-digit class (0-5), so a
# quadrant-assignment miss shows up as its own bucket instead of being silently absorbed into
# whichever last digit the model happened to guess (or, the opposite failure mode, silently
# scored as "correct" just because the last digit matched while the tooth was attributed to the
# wrong tooth position entirely).
WRONG_QUADRANT = 6


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



# ---------------------------------------------------------------------------
# Tooth-level / arch-level metric bucketing, shared across all three models' eval scripts
# ---------------------------------------------------------------------------
def build_missing_count_buckets(per_jaw, full_arch_size=12):
    """{(jaw, image_name): missing_count} for every arch in per_jaw - the same bucketing
    compute_missing_teeth_metrics uses internally, exposed standalone so a caller can reuse it to
    bucket per-TOOTH data too (see compute_tooth_level_metrics_by_bucket)."""
    arch_true_sets = {}
    for jaw in ("lower", "upper"):
        d = per_jaw[jaw]
        for fdi_number, image_name in zip(d["y_fdi_number"], d["y_image_name"]):
            arch_true_sets.setdefault((jaw, image_name), set()).add(fdi_number)
    return {k: max(full_arch_size - len(v), 0) for k, v in arch_true_sets.items()}


def build_complexity_buckets(per_jaw, complexity_by_image, merge_classes=None):
    """{(jaw, image_name): complexity_class} for every arch in per_jaw with a known complexity
    label - the same bucketing compute_arch_metrics_by_complexity uses internally. merge_classes,
    if given (e.g. {2, 3}), collapses every listed class onto a single "II+III"-style bucket key
    (the merged class values joined with '+'), for a pooled higher-complexity population instead
    of separate per-class ones."""
    labels_by_class = {1: "I", 2: "II", 3: "III"}
    merged_key = "+".join(labels_by_class.get(c, str(c)) for c in sorted(merge_classes)) if merge_classes else None

    buckets = {}
    for jaw in ("lower", "upper"):
        d = per_jaw[jaw]
        for image_name in d["y_image_name"]:
            key = (jaw, image_name)
            if key in buckets:
                continue
            c = complexity_by_image.get(image_name)
            if c is None:
                continue
            buckets[key] = merged_key if (merge_classes and c in merge_classes) else c
    return buckets


def compute_tooth_level_metrics_by_bucket(per_jaw, arch_bucket, pred_key, tens_pred_key="y_tens_pred"):
    """Per-TOOTH-level Accuracy/Precision/Recall/F1 (same convention as compute_tooth_number_metrics:
    gated by tens, so a tooth only counts as correct when both its predicted last digit and
    predicted FDI tens digit match; macro P/R/F1 over the 6 last-digit classes), computed
    separately per bucket instead of pooled across the whole test set. arch_bucket:
    {(jaw, image_name): bucket_key}, as built by build_missing_count_buckets /
    build_complexity_buckets (or a caller's own remapping of either, e.g. collapsing every
    missing_count >= 1 onto one "missing" key) - a tooth whose arch isn't in arch_bucket is
    skipped. Used for the per-tooth half of Table 1/Table 2's 8-column (per-tooth + per-arch)
    format, so the per-tooth and per-arch columns describe the exact same arch population."""
    bucket_true, bucket_pred = {}, {}
    for jaw in ("lower", "upper"):
        d = per_jaw[jaw]
        gated_pred = gate_by_tens(d[pred_key], d[tens_pred_key], d["y_fdi_number"])
        for true_digit, pred_digit, image_name in zip(d["y_true"], gated_pred, d["y_image_name"]):
            key = arch_bucket.get((jaw, image_name))
            if key is None:
                continue
            bucket_true.setdefault(key, []).append(true_digit)
            bucket_pred.setdefault(key, []).append(pred_digit)

    digit_labels = list(range(6))
    results = {}
    for key in bucket_true:
        yt, yp = bucket_true[key], bucket_pred[key]
        yt_arr, yp_arr = np.asarray(yt), np.asarray(yp)
        # Specificity (macro, one-vs-rest over the 6 last-digit classes): TN/(TN+FP) per class,
        # averaged - not provided by sklearn directly (it only has recall_score's pos_label /
        # multi-class recall, which is sensitivity, not specificity), so computed by hand from the
        # same gated true/pred arrays used for the other four metrics above.
        specs = []
        for c in digit_labels:
            tn = np.sum((yt_arr != c) & (yp_arr != c))
            fp = np.sum((yt_arr != c) & (yp_arr == c))
            specs.append(tn / (tn + fp) if (tn + fp) else float("nan"))
        results[key] = {
            "n_teeth": len(yt),
            "accuracy": accuracy_score(yt, yp),
            "precision": precision_score(yt, yp, labels=digit_labels, average="macro", zero_division=0),
            "recall": recall_score(yt, yp, labels=digit_labels, average="macro", zero_division=0),
            "specificity": float(np.nanmean(specs)),
            "f1": f1_score(yt, yp, labels=digit_labels, average="macro", zero_division=0),
        }
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
    functions/features/complexity_labels.py's load_complexity_labels) instead of missing-tooth status -
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
