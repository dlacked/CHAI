"""
CSV builder for Comp3 (paper label; ablation of Baseline = functions/features/main_baseline.py /
coords_baseline.py, "CHAI (Ours)"): up-to-tens-digit (12-way: this jaw's 2 quadrants x 6 digits,
same class12 scheme as the old E/Comp2 ablation - see main_e12way.py), evaluated with NO Hungarian
post-process (raw per-tooth argmax stands as the final answer) - isolates "how much does skipping
the separate geometric-tens-assignment + Hungarian pipeline cost."

RENAMED 2026-08-31 from main_comp2.py: this used to also build Comp2's CSV (last-digit-only,
6-way, sharing Comp3's exact coordinates - Baseline's Y-flip kept, X-mirror dropped). Comp2 was
then redefined to fully-raw coordinates (no transform at all, incl. no Y-flip) with separate
per-jaw models instead of pooling - see functions/features/main_comp2.py (the NEW file at that
name, writing ResNet/tooth/csv_comp2/) - so this file (and csv_comp3/) is now Comp3-only. The
`fdi_last_digit` column below is vestigial (Comp3 only reads class12) - kept rather than removed,
harmless either way.

Uses coords_comp3.py (Baseline's Y-flip kept, X-mirror dropped - see that module's docstring for
why Comp3 needs un-mirrored coordinates), combined-jaws pooling, and the same
PCA-for-arch-ordering-only setup as Baseline.

main_baseline.py and its outputs (ResNet/tooth/csv_baseline/) are left completely alone - this
writes to ResNet/tooth/csv_comp3/ instead.

Usage:
    .venv/Scripts/python.exe functions/features/main_comp3.py [--force]
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

from coords_comp3 import get_normalized_coords_comp3
from fdi_last import get_centroid
from theta import calculate_pca_rotation, compute_arch_order

# This jaw's two tens values, in local-index order (0, 1) - same convention as main_e12way.py's
# class12, so Comp3's predictions carry over that scheme unchanged, just now from a pooled model.
JAW_TENS = {"upper": (1, 2), "lower": (3, 4)}


def process_jaw(jaw, split, dataset_dir, results_data):
    json_dir = Path(dataset_dir) / split / "labels_json" / jaw
    image_dir = Path(dataset_dir) / split / "images" / jaw

    if not json_dir.exists():
        print(f"Error: JSON labels directory {json_dir} does not exist.")
        return

    print(f"\n--- Starting GT feature extraction (no-mirror, pooled, Comp3) for {jaw.upper()} jaw ({split}) ---")
    print(f"JSON label directory: {json_dir}")

    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not json_files:
        print(f"No JSON label files found in {json_dir}")
        return

    tens_local_index = {JAW_TENS[jaw][0]: 0, JAW_TENS[jaw][1]: 1}

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
                if (num // 10) not in JAW_TENS[jaw]:
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

            # PCA here is only for compute_arch_order's row ordering - not used for coordinates,
            # same as main_baseline.py (verified robust to a missing tooth, see project notes).
            centroids = [item["centroid"] for item in segments_data]
            mean_pt, angle = calculate_pca_rotation(centroids)
            order = compute_arch_order(centroids, mean_pt, angle, jaw == "upper")
            sorted_segments = [segments_data[i] for i in order]

            for seg_data in sorted_segments:
                fdi_number = seg_data["fdi_number"]
                fdi_tens = fdi_number // 10
                fdi_digit = fdi_number % 10

                x1, y1, x2, y2 = get_normalized_coords_comp3(
                    seg_data["poly"], img_size, jaw
                )
                class12 = tens_local_index[fdi_tens] * 6 + (fdi_digit - 1)

                results_data.append([
                    image_name,
                    jaw,
                    x1,
                    y1,
                    x2,
                    y2,
                    fdi_digit,
                    class12,
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
    output_dir = project_root / "ResNet" / "tooth" / "csv_comp3"

    parser = argparse.ArgumentParser(
        description="Extract coordinates with no PCA rotation and no X-mirror (Y-flip kept), "
                     "combined-jaws CSV for Comp3 (class12 label; fdi_last_digit column is "
                     "vestigial, kept for convenience)."
    )
    parser.add_argument("--dataset_dir", type=str, default=str(dataset_dir))
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    splits = ["train", "val", "test"]
    jaws = ["lower", "upper"]
    headers = ["image_name", "jaw", "x1", "y1", "x2", "y2", "fdi_last_digit", "class12", "fdi_tens", "fdi_number"]

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

    print("\nNo-mirror pooled (Comp3) feature extraction completed successfully.")


if __name__ == "__main__":
    main()
