import csv
import json
import argparse
from pathlib import Path
from collections import OrderedDict

import torch

COMPLEXITY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = COMPLEXITY_DIR.parent.parent

MAX_TEETH_PER_ARCH = 12  # matches the 12 FDI slots (1-6) used per side/jaw everywhere else
GEOM_FIELDS = ("x1", "y1", "x2", "y2", "theta")


def load_complexity_labels(dataset_dir, split):
    meta_path = Path(dataset_dir) / split / "metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        info = json.load(f)["info"]
    return {Path(e["image_filepath"]).name: e["complexity"] for e in info}


def build_split(jaw, split, dataset_dir, csv_dir):
    csv_path = Path(csv_dir) / f"{jaw}_features_{split}.csv"
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}. Skipping {jaw}/{split}.")
        return []

    complexity_by_image = load_complexity_labels(dataset_dir, split)

    # CSV rows are written image-by-image, arch-ordered left to right (functions/features/main.py),
    # so consecutive rows sharing image_name are one arch.
    groups = OrderedDict()
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
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
    csv_dir_default = PROJECT_ROOT / "ResNet" / "tooth" / "csv"
    cache_dir_default = COMPLEXITY_DIR / "cache"

    parser = argparse.ArgumentParser(description="Build offline arch-geometry cache for the ViT arch-complexity classifier.")
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
            print(f"\nBuilding {jaw}/{split} arch sequences...")
            sequences = build_split(jaw, split, args.dataset_dir, args.csv_dir)
            if not sequences:
                continue
            print(f"  {len(sequences)} arches.")
            torch.save(sequences, out_path)
            print(f"  Saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
