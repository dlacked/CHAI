"""
This is Baseline - the main CHAI model ("CHAI (Ours)" in the paper tables) - combining two
changes at once vs. the old production/variant-A scheme (main.py, superseded and moved to
backups/superseded_20260831/ along with its outputs - see project notes for the full reasoning,
which starts from tight-crop normalization and per-feature fixes being found to still depend on
the same fragile PCA axis):

  1. No PCA coordinate rotation - x1/y1/x2/y2 come straight from the polygon's raw image-space
     bbox (coords_baseline.get_normalized_coords_baseline), mirrored/flipped by image geometry alone
     (image width/2 for the X mirror, image height for the Y flip - see that module's docstring).
     PCA (calculate_pca_rotation) is still computed here, but ONLY to feed compute_arch_order
     (Held-Karp row ordering) - that step was verified robust to a missing tooth (leave-one-out
     test: 0/3000 arches saw their relative order among the remaining teeth change), unlike the
     coordinate rotation and theta, both of which were shown to swing by ~13deg on average when a
     single posterior molar is missing.
  2. No theta feature - it's `arctan2(rotated_y, rotated_x)` against the same PCA axis, so it
     inherited the coordinate rotation's fragility almost exactly (leave-one-out test: ~14.5deg
     average shift on a neighboring tooth when a posterior molar is missing, comparable to the
     axis's own ~13deg swing). Not worth keeping once the mechanism producing it is this fragile.

Also combines both jaws into ONE CSV per split (the old B/C attempts at this, main_nomirror.py /
main_unified_mirror.py, are superseded - see coords_baseline.py's docstring for why). The X mirror
rule itself is unchanged from production (tens in (2, 3)) - what's new is the Y flip for the
whole lower jaw (see coords_baseline.py's docstring for why both are needed to make all 4 quadrants
land in one consistent coordinate convention for a single pooled model).

Writes to ResNet/tooth/csv_baseline/.

Usage:
    .venv/Scripts/python.exe functions/features/main_baseline.py [--force]
"""
import sys
import os
import csv
import argparse
import glob
import json
from pathlib import Path
from PIL import Image

features_dir = Path(__file__).resolve().parent
if str(features_dir) not in sys.path:
    sys.path.append(str(features_dir))

from coords_baseline import get_normalized_coords_baseline
from fdi_last import get_centroid
from theta import calculate_pca_rotation, compute_arch_order


def process_jaw(jaw, split, dataset_dir, results_data):
    """Appends this jaw's rows (image_name, jaw, x1, y1, x2, y2, fdi_last_digit, fdi_tens,
    fdi_number) onto results_data in place. PCA is computed per image only to drive
    compute_arch_order's row ordering - it never touches the coordinates themselves (see this
    module's docstring)."""
    json_dir = Path(dataset_dir) / split / "labels_json" / jaw
    image_dir = Path(dataset_dir) / split / "images" / jaw

    if not json_dir.exists():
        print(f"Error: JSON labels directory {json_dir} does not exist.")
        return

    print(f"\n--- Starting GT feature extraction (no-rotation, pooled) for {jaw.upper()} jaw ({split}) ---")
    print(f"JSON label directory: {json_dir}")

    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not json_files:
        print(f"No JSON label files found in {json_dir}")
        return

    for json_path in json_files:
        filename_no_ext = Path(json_path).stem

        image_name = None
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".webp"]:
            test_img_path = image_dir / f"{filename_no_ext}{ext}"
            if test_img_path.exists():
                image_name = f"{filename_no_ext}{ext}"
                img_path = test_img_path
                break

        if image_name is None:
            print(f"Warning: Image file not found for {filename_no_ext}. Skipping.")
            continue

        try:
            with Image.open(img_path) as img:
                img_size = img.size

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            teeth = data.get("tooth", [])
            if not teeth:
                continue

            segments_data = []
            for t in teeth:
                num = t.get("teeth_num")
                seg = t.get("segmentation", [])
                if not seg or num is None:
                    continue

                if isinstance(seg[0], list) and len(seg[0]) == 2:
                    poly = seg
                else:
                    poly = [[seg[i], seg[i + 1]] for i in range(0, len(seg), 2)]

                c = get_centroid(poly)
                segments_data.append({
                    "fdi_number": num,
                    "poly": poly,
                    "centroid": c
                })

            if not segments_data:
                continue

            # PCA here is only for compute_arch_order's row ordering - not used for coordinates.
            centroids = [item["centroid"] for item in segments_data]
            mean_pt, angle = calculate_pca_rotation(centroids)
            order = compute_arch_order(centroids, mean_pt, angle, jaw == "upper")
            sorted_segments = [segments_data[i] for i in order]

            for seg_data in sorted_segments:
                fdi_number = seg_data["fdi_number"]
                if fdi_number is None or fdi_number == -1:
                    continue
                fdi_tens = fdi_number // 10
                fdi_digit = fdi_number % 10
                if fdi_tens not in (1, 2, 3, 4) or not (1 <= fdi_digit <= 6):
                    continue

                x1, y1, x2, y2 = get_normalized_coords_baseline(
                    seg_data["poly"], img_size, fdi_number, jaw
                )

                results_data.append([
                    image_name,
                    jaw,
                    x1,
                    y1,
                    x2,
                    y2,
                    fdi_digit,
                    fdi_tens,
                    fdi_number
                ])

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing JSON {json_path}: {e}")
            continue


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_dir = project_root.parent / "dataset"
    output_dir = project_root / "ResNet" / "tooth" / "csv_baseline"

    parser = argparse.ArgumentParser(
        description="Extract coordinates/FDI features with no PCA rotation, a unified (10s-"
                     "referenced) X mirror + jaw Y-flip, and a combined (both jaws) CSV per "
                     "split, 6-way digit label, no theta column - this is Baseline, CHAI (Ours)."
    )
    parser.add_argument("--dataset_dir", type=str, default=str(dataset_dir),
                         help="Path to the dataset directory")
    parser.add_argument("--force", action="store_true",
                         help="Force feature extraction even if CSV files exist")

    args = parser.parse_args()

    splits = ["train", "val", "test"]
    jaws = ["lower", "upper"]
    headers = ["image_name", "jaw", "x1", "y1", "x2", "y2", "fdi_last_digit", "fdi_tens", "fdi_number"]

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        csv_path = output_dir / f"features_{split}.csv"
        if csv_path.exists() and not args.force:
            print(f"CSV file already exists: {csv_path.resolve()}. Skipping (use --force to overwrite).")
            continue

        results_data = []
        for jaw in jaws:
            process_jaw(jaw, split, args.dataset_dir, results_data)

        if not results_data:
            print(f"No features extracted for split {split}.")
            continue

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(results_data)
            print(f"Success: Saved features to {csv_path.resolve()}")
        except Exception as e:
            print(f"Error writing CSV file {csv_path}: {e}")

    print("\nNo-rotation, pooled feature extraction completed successfully.")


if __name__ == "__main__":
    main()
