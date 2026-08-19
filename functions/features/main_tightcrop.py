"""
Ablation counterpart to main.py: builds the SAME training CSVs (image_name, x1, y1, x2, y2,
theta, fdi_last_digit, fdi_number) but with each tooth's x1/y1/x2/y2 normalized against its own
arch's tight-crop bounds (coords_tightcrop.get_normalized_coords_tightcrop) instead of the full
source photo's width/height (coords.get_normalized_coords). theta/PCA/arch-ordering are
untouched - reused directly from theta.py, since those don't depend on image framing at all.

main.py and its outputs (ResNet/tooth/csv/) are left completely alone - this writes to
ResNet/tooth/csv_tightcrop/ instead, so both normalization schemes' training data exist side by
side for the planned model comparison (see project memory: does normalization choice actually
matter, framed as a likely paper-review question).

Usage:
    .venv/Scripts/python.exe functions/features/main_tightcrop.py [--force]
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

from coords_tightcrop import get_normalized_coords_tightcrop, compute_arch_tight_bbox
from fdi_last import get_centroid
from theta import get_theta_values, calculate_pca_rotation, compute_arch_order


def process_jaw(jaw, split, dataset_dir, output_csv):
    json_dir = Path(dataset_dir) / split / "labels_json" / jaw
    image_dir = Path(dataset_dir) / split / "images" / jaw

    if not json_dir.exists():
        print(f"Error: JSON labels directory {json_dir} does not exist.")
        return

    print(f"\n--- Starting GT feature extraction (tight-crop norm) for {jaw.upper()} jaw ({split}) ---")
    print(f"JSON label directory: {json_dir}")

    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not json_files:
        print(f"No JSON label files found in {json_dir}")
        return

    results_data = []

    for json_path in json_files:
        filename_no_ext = Path(json_path).stem

        image_name = None
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".webp"]:
            test_img_path = image_dir / f"{filename_no_ext}{ext}"
            if test_img_path.exists():
                image_name = f"{filename_no_ext}{ext}"
                break

        if image_name is None:
            print(f"Warning: Image file not found for {filename_no_ext}. Skipping.")
            continue

        try:
            # Unlike main.py, img_size is never used - this pipeline normalizes against the
            # arch's own tight crop, not the photo's dimensions - so the image itself doesn't
            # even need to be opened here.
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

            centroids = [item["centroid"] for item in segments_data]
            mean_pt, angle = calculate_pca_rotation(centroids)

            pca_results = get_theta_values(centroids, mean_pt, angle)
            for i, item in enumerate(segments_data):
                pca_info = pca_results[i]
                item["rotated_x"] = pca_info["rotated_x"]
                item["theta"] = pca_info["angle_norm"]

            order = compute_arch_order(centroids, mean_pt, angle, jaw == "upper")
            sorted_segments = [segments_data[i] for i in order]

            # The one real addition vs. main.py: one shared tight-crop bbox for the whole arch,
            # computed once (not per tooth) - see compute_arch_tight_bbox's docstring for why its
            # X extent has to be computed in the same mirrored frame as each tooth's own X below.
            crop_bbox = compute_arch_tight_bbox(
                [(seg["poly"], seg["fdi_number"]) for seg in sorted_segments], mean_pt, angle
            )

            for seg_data in sorted_segments:
                fdi_number = seg_data["fdi_number"]
                fdi_digit = fdi_number % 10

                x1, y1, x2, y2 = get_normalized_coords_tightcrop(
                    seg_data["poly"], mean_pt, angle, crop_bbox, fdi_number
                )

                theta_val_raw = seg_data["theta"]
                if fdi_number != -1 and fdi_number is not None:
                    tens = fdi_number // 10
                    if tens in (2, 3):
                        if theta_val_raw >= 0:
                            theta_val_raw = 1.0 - theta_val_raw
                        else:
                            theta_val_raw = -1.0 - theta_val_raw

                theta_val = f"{theta_val_raw:.6f}"

                results_data.append([
                    image_name,
                    x1,
                    y1,
                    x2,
                    y2,
                    theta_val,
                    fdi_digit,
                    fdi_number
                ])

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing JSON {json_path}: {e}")
            continue

    if not results_data:
        print(f"No features extracted for {jaw.upper()} jaw.")
        return

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    headers = ["image_name", "x1", "y1", "x2", "y2", "theta", "fdi_last_digit", "fdi_number"]

    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(results_data)
        print(f"Success: Saved features to {output_csv.resolve()}")
    except Exception as e:
        print(f"Error writing CSV file {output_csv}: {e}")


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_dir = project_root.parent / "dataset"
    output_dir = project_root / "ResNet" / "tooth" / "csv_tightcrop"

    parser = argparse.ArgumentParser(
        description="Extract coordinates/theta/FDI features, normalized against each arch's own "
                     "tight crop instead of the full photo (ablation counterpart to main.py)."
    )
    parser.add_argument("--dataset_dir", type=str, default=str(dataset_dir),
                         help="Path to the dataset directory")
    parser.add_argument("--force", action="store_true",
                         help="Force feature extraction even if CSV files exist")

    args = parser.parse_args()

    splits = ["train", "val"]
    jaws = ["lower", "upper"]

    for split in splits:
        for jaw in jaws:
            csv_path = output_dir / f"{jaw}_features_{split}.csv"
            if csv_path.exists() and not args.force:
                print(f"CSV file already exists: {csv_path.resolve()}. Skipping (use --force to overwrite).")
                continue
            process_jaw(jaw, split, args.dataset_dir, csv_path)

    print("\nTight-crop-normalized feature extraction completed successfully.")


if __name__ == "__main__":
    main()
