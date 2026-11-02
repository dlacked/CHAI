"""
Reruns the Yoon et al. reproduction's test-set inference (same Cascade R-CNN end-to-end detect+
classify pipeline as evaluate.py) to rebuild per_jaw, then additionally computes the per-TOOTH
metrics (accuracy/precision/recall/F1, gated by tens - see functions/eval_metrics.py's
compute_tooth_level_metrics_by_bucket) needed for access.tex's 8-column Table 1 (missing-tooth
subset) and Table 2 (Class II+III merged subset) restructure, alongside the existing per-ARCH
metrics those tables already report. Kept as a separate script rather than edited into evaluate.py
so the original file (and its own saved results) stays untouched.

per_jaw itself is also pickled for reuse, so a further table iteration doesn't need to re-run
Cascade R-CNN inference over the whole test set again.

Usage:
    .venv/Scripts/python.exe comparison/yoon/evaluate_table_restructure.py [checkpoint_path]
"""
import json
import pickle
import sys
from pathlib import Path

YOON_DIR = Path(__file__).resolve().parent
if str(YOON_DIR) not in sys.path:
    sys.path.insert(0, str(YOON_DIR))

from evaluate import find_checkpoint, evaluate_split, PROJECT_ROOT, DATASET_DIR  # noqa: E402

RESULTS_DIR = YOON_DIR / "results"


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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "yoon_per_jaw.pkl", "wb") as f:
        pickle.dump(per_jaw, f)
    print(f"Saved raw per-tooth predictions to {RESULTS_DIR / 'yoon_per_jaw.pkl'}")

    tooth_module = load_module("chai_tooth_graph_yt", PROJECT_ROOT / "functions" / "eval_metrics.py")
    complexity_build = load_module(
        "vit_complexity_build_yt", PROJECT_ROOT / "functions" / "features" / "complexity_labels.py")
    complexity_by_image = complexity_build.load_complexity_labels(DATASET_DIR, "test")

    missing_arch = tooth_module.compute_missing_vs_complete_metrics(
        per_jaw, pred_key="y_pred", prob_field="prob", tens_pred_key="y_tens_pred"
    )
    complexity_arch = tooth_module.compute_arch_metrics_by_complexity(
        per_jaw, complexity_by_image, pred_key="y_pred", prob_field="prob", tens_pred_key="y_tens_pred"
    )

    missing_buckets = tooth_module.build_missing_count_buckets(per_jaw)
    missing_bucket_2way = {k: ("missing" if v >= 1 else "complete") for k, v in missing_buckets.items()}
    missing_tooth = tooth_module.compute_tooth_level_metrics_by_bucket(
        per_jaw, missing_bucket_2way, pred_key="y_pred", tens_pred_key="y_tens_pred"
    )

    complexity_buckets_merged = tooth_module.build_complexity_buckets(
        per_jaw, complexity_by_image, merge_classes={2, 3}
    )
    complexity_tooth_merged = tooth_module.compute_tooth_level_metrics_by_bucket(
        per_jaw, complexity_buckets_merged, pred_key="y_pred", tens_pred_key="y_tens_pred"
    )

    print("\n=== Missing-tooth subset (>=1 missing) ===")
    print("Per-arch:", missing_arch["missing"])
    print("Per-tooth:", missing_tooth.get("missing"))

    print("\n=== Class II+III merged subset ===")
    print("Per-arch Class II:", complexity_arch.get(2))
    print("Per-arch Class III:", complexity_arch.get(3))
    print("Per-tooth Class II+III merged:", complexity_tooth_merged.get("II+III"))

    out = {
        "checkpoint": str(checkpoint),
        "missing_arch": missing_arch,
        "complexity_arch": complexity_arch,
        "missing_tooth": missing_tooth,
        "complexity_tooth_merged": complexity_tooth_merged,
    }
    with open(RESULTS_DIR / "yoon_table_restructure.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_DIR / 'yoon_table_restructure.json'}")


if __name__ == "__main__":
    main()
