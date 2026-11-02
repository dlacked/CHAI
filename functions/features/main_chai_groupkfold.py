"""
GroupKFold variant of main_chai.py: builds ONE combined per-tooth feature CSV spanning every
image in train+val+test (71,520 images), tagged with each image's new CV role from
cv_split_assignment.csv/.pkl (role="test" or role="pool"+fold=0..9) instead of writing three
separate features_{train,val,test}.csv files keyed by the OLD fixed split. Training scripts can
then filter this one CSV by role/fold to build each fold's train/val sets, and by role="test" for
the fixed held-out test set - see cv_split_assignment.csv's docstring (finalize_group_kfold_split.py)
for how the split itself was built.

Reuses main_chai.py's own per-tooth extraction logic (coords_chai, same class12 scheme) unchanged;
the only difference is iterating over all three dataset_dir/{split}/ trees in one pass and
attaching group_id/role/fold columns looked up from the CV assignment table, instead of one
dataset_dir/{split}/ tree per output CSV.

Usage:
    .venv/Scripts/python.exe functions/features/main_chai_groupkfold.py [--force]
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

from coords_chai import get_normalized_coords_chai
from fdi_last import get_centroid
from theta import calculate_pca_rotation, compute_arch_order

JAW_TENS = {"upper": (1, 2), "lower": (3, 4)}


def load_cv_assignment(root):
    import pickle
    with open(root / "cv_split" / "cv_split_assignment.pkl", "rb") as f:
        assignment = pickle.load(f)
    lookup = {}
    for orig_split, jaw, image_name, group_id, role, fold in assignment:
        lookup[(orig_split, jaw, image_name)] = (group_id, role, fold)
    return lookup


def process_jaw(jaw, split, dataset_dir, cv_lookup, results_data):
    json_dir = Path(dataset_dir) / split / "labels_json" / jaw
    image_dir = Path(dataset_dir) / split / "images" / jaw

    if not json_dir.exists():
        print(f"Error: JSON labels directory {json_dir} does not exist.")
        return

    print(f"\n--- GT feature extraction (GroupKFold, CHAI) for {jaw.upper()} jaw ({split}) ---")

    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not json_files:
        print(f"No JSON label files found in {json_dir}")
        return

    tens_local_index = {JAW_TENS[jaw][0]: 0, JAW_TENS[jaw][1]: 1}

    n_missing_cv = 0
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

        cv_key = (split, jaw, image_name)
        if cv_key not in cv_lookup:
            n_missing_cv += 1
            continue
        group_id, role, fold = cv_lookup[cv_key]

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
                segments_data.append({"fdi_number": num, "poly": poly, "centroid": c})

            if not segments_data:
                continue

            centroids = [item["centroid"] for item in segments_data]
            mean_pt, angle = calculate_pca_rotation(centroids)
            order = compute_arch_order(centroids, mean_pt, angle, jaw == "upper")
            sorted_segments = [segments_data[i] for i in order]

            for seg_data in sorted_segments:
                fdi_number = seg_data["fdi_number"]
                fdi_tens = fdi_number // 10
                fdi_digit = fdi_number % 10

                x1, y1, x2, y2 = get_normalized_coords_chai(seg_data["poly"], img_size, jaw)
                class12 = tens_local_index[fdi_tens] * 6 + (fdi_digit - 1)

                results_data.append([
                    image_name, jaw, x1, y1, x2, y2, fdi_digit, class12, fdi_tens, fdi_number,
                    split, group_id, role, fold,
                ])

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing JSON {json_path}: {e}")
            continue

    if n_missing_cv:
        print(f"WARNING: {n_missing_cv} images in {split}/{jaw} had no CV assignment (unexpected).")


def main():
    _bootstrap_root = Path(__file__).resolve().parent.parent.parent
    if str(_bootstrap_root) not in sys.path:
        sys.path.append(str(_bootstrap_root))
    from config import PathConfig
    project_root = PathConfig.ROOT
    dataset_dir = project_root.parent / "dataset"
    output_path = project_root / "ResNet" / "csv" / "features_groupkfold.csv"

    parser = argparse.ArgumentParser(
        description="Build one combined per-tooth feature CSV (train+val+test) tagged with "
                     "GroupKFold CV role/fold, for CHAI."
    )
    parser.add_argument("--dataset_dir", type=str, default=str(dataset_dir))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if output_path.exists() and not args.force:
        print(f"CSV file already exists: {output_path}. Skipping (use --force to overwrite).")
        return

    cv_lookup = load_cv_assignment(project_root)
    print(f"Loaded CV assignment for {len(cv_lookup)} images.")

    splits = ["train", "val", "test"]
    jaws = ["lower", "upper"]
    headers = ["image_name", "jaw", "x1", "y1", "x2", "y2", "fdi_last_digit", "class12",
               "fdi_tens", "fdi_number", "orig_split", "group_id", "cv_role", "cv_fold"]

    results_data = []
    for split in splits:
        for jaw in jaws:
            process_jaw(jaw, split, args.dataset_dir, cv_lookup, results_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results_data)
    print(f"\nSuccess: Saved {len(results_data)} tooth rows to {output_path}")


if __name__ == "__main__":
    main()
