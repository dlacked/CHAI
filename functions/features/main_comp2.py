"""
CSV builder for Comp2 (paper label; ablation of Baseline = functions/features/main_baseline.py /
coords_baseline.py, "CHAI (Ours)"): last-digit-only (6-way), fully raw coordinates (no PCA
rotation, no X-mirror, no Y-flip - see coords_comp2.py's docstring) - isolates "what happens if
we skip every geometric coordinate transform entirely," as opposed to Comp3 (main_comp3.py) which
only isolates the X-mirror step.

REDEFINED 2026-08-31: previously this file (then producing csv_comp2/) built the SAME
coordinates Comp3 uses (Y-flip kept, X-mirror dropped, pooled). That definition is now
Comp3-only (renamed to main_comp3.py / coords_comp3.py / csv_comp3/) - Comp2 moved to fully-raw
coordinates instead. Since there's no Y-flip left to align upper/lower jaw coordinate ranges, this
still writes ONE combined CSV per split (with a `jaw` column, like csv_baseline/) but
ResNet/tooth/train_comp2.py filters by jaw and trains lower/upper as two separate models - keeping
one CSV (filtered at train time) avoids duplicating this whole extraction script per jaw.

Still uses PCA for compute_arch_order's row ordering only (same as every other main_*.py here -
verified robust to a missing tooth, see project notes) - never for the coordinates themselves.

main_baseline.py/main_comp3.py and their outputs are left completely alone - this writes to
ResNet/tooth/csv_comp2/ instead.

Usage:
    .venv/Scripts/python.exe functions/features/main_comp2.py [--force]
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

from coords_comp2 import get_normalized_coords_comp2
from fdi_last import get_centroid
from theta import calculate_pca_rotation, compute_arch_order


def process_jaw(jaw, split, dataset_dir, results_data):
    json_dir = Path(dataset_dir) / split / "labels_json" / jaw
    image_dir = Path(dataset_dir) / split / "images" / jaw

    if not json_dir.exists():
        print(f"Error: JSON labels directory {json_dir} does not exist.")
        return

    print(f"\n--- Starting GT feature extraction (raw, no-transform, Comp2) for {jaw.upper()} jaw ({split}) ---")
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

                x1, y1, x2, y2 = get_normalized_coords_comp2(seg_data["poly"], img_size)

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
    output_dir = project_root / "ResNet" / "tooth" / "csv_comp2"

    parser = argparse.ArgumentParser(
        description="Extract fully raw coordinates (no PCA, no X-mirror, no Y-flip) for Comp2, "
                     "one combined (both-jaws, `jaw` column) CSV per split - filtered by jaw at "
                     "train time since there's no per-jaw model."
    )
    parser.add_argument("--dataset_dir", type=str, default=str(dataset_dir))
    parser.add_argument("--force", action="store_true")

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

    print("\nRaw (no-transform) Comp2 feature extraction completed successfully.")


if __name__ == "__main__":
    main()
