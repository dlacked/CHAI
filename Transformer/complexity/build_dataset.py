import csv
import json
import argparse
from pathlib import Path
from collections import OrderedDict

import torch

COMPLEXITY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = COMPLEXITY_DIR.parent.parent

MAX_TEETH_PER_ARCH = 12  # matches the 12 FDI slots (1-6) used per side/jaw everywhere else
GEOM_FIELDS = ("x1", "y1", "x2", "y2")  # Main Arch Complexity data-build script (no theta, no
                                          # PCA, no X-mirror - see csv_dir_default in main() below)
                                          # - official model as of 2026-08-31. The theta-included
                                          # ablation baseline (build_dataset_theta.py/train_theta.py)
                                          # still reads the old PCA-rotated per-jaw CSVs, so it
                                          # isolates theta ALONE - it is no longer a clean apples-
                                          # to-apples comparison against this file after this
                                          # coordinate-scheme change; treat it as a separate,
                                          # earlier ablation rather than "same model minus theta."


def load_complexity_labels(dataset_dir, split):
    meta_path = Path(dataset_dir) / split / "metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        info = json.load(f)["info"]
    return {Path(e["image_filepath"]).name: e["complexity"] for e in info}


def build_split(jaw, split, dataset_dir, csv_dir):
    # csv_comp2 is ONE combined (both-jaws) file per split, not per-jaw files like the old
    # per-jaw PCA-rotated CSVs this used to read - filter to this jaw's rows.
    csv_path = Path(csv_dir) / f"features_{split}.csv"
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}. Skipping {jaw}/{split}.")
        return []

    complexity_by_image = load_complexity_labels(dataset_dir, split)

    # main_comp2.py writes ALL of one jaw's rows (each image-by-image, arch-ordered left to
    # right) before starting the other jaw's, so filtering by jaw here still leaves each image's
    # rows contiguous and in Held-Karp arch order - same grouping assumption as before.
    groups = OrderedDict()
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["jaw"] != jaw:
                continue
            groups.setdefault(row["image_name"], []).append(row)

    sequences = []
    skipped_long = 0
    skipped_no_label = 0
    for image_name, rows in groups.items():
        complexity = complexity_by_image.get(image_name)
        if complexity is None:
            skipped_no_label += 1
            continue

        if len(rows) > MAX_TEETH_PER_ARCH:
            skipped_long += 1
            rows = rows[:MAX_TEETH_PER_ARCH]

        geom = torch.tensor(
            [[float(row[f]) for f in GEOM_FIELDS] for row in rows],
            dtype=torch.float32,
        )

        sequences.append({
            "image_name": image_name,
            "geom": geom,
            "complexity": torch.tensor(complexity - 1, dtype=torch.long),  # 0-indexed for CrossEntropyLoss
        })

    if skipped_long:
        print(f"Warning: {skipped_long} arches in {jaw}/{split} had more than {MAX_TEETH_PER_ARCH} teeth; truncated.")
    if skipped_no_label:
        print(f"Warning: {skipped_no_label} arches in {jaw}/{split} had no complexity label; skipped.")

    return sequences


def main():
    dataset_dir_default = PROJECT_ROOT.parent / "dataset"
    # csv_comp3 (functions/features/main_comp3.py): no PCA rotation, no X-mirror, Y-flip kept -
    # deliberately NOT csv_baseline, whose X-mirror folds one side of the arch onto the other
    # (good for the 6-way ResNet digit classifier, but destroys the whole-arch left-right shape
    # Arch Complexity actually needs to see) and NOT the old archived per-jaw PCA-rotated CSVs
    # (PCA axis swings ~13deg on a single missing tooth - actively harmful for a model whose
    # whole point is judging complexity in exactly the arches most likely to have one). NOTE:
    # csv_comp2 is NOT this - as of 2026-08-31 Comp2 means fully-raw (no Y-flip either) coords,
    # a different scheme than what Arch Complexity needs (see coords_comp2.py's docstring).
    csv_dir_default = PROJECT_ROOT / "ResNet" / "tooth" / "csv_comp3"
    cache_dir_default = COMPLEXITY_DIR / "cache"

    parser = argparse.ArgumentParser(description="Build offline arch-geometry cache (no theta, no PCA, no X-mirror) for the Arch Complexity Transformer.")
    parser.add_argument("--dataset_dir", type=str, default=str(dataset_dir_default))
    parser.add_argument("--csv_dir", type=str, default=str(csv_dir_default))
    parser.add_argument("--cache_dir", type=str, default=str(cache_dir_default))
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache files")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for jaw in ("lower", "upper"):
        for split in ("train", "val", "test"):
            out_path = cache_dir / f"{jaw}_{split}.pt"
            if out_path.exists() and not args.force:
                print(f"Cache already exists: {out_path}. Skipping (use --force to overwrite).")
                continue
            print(f"\nBuilding {jaw}/{split} arch sequences (no theta)...")
            sequences = build_split(jaw, split, args.dataset_dir, args.csv_dir)
            if not sequences:
                continue
            print(f"  {len(sequences)} arches.")
            torch.save(sequences, out_path)
            print(f"  Saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
