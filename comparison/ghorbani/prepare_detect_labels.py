"""
Converts CHAI's per-tooth GT polygon labels (dataset/{split}/labels_json/{jaw}/*.json) into
plain YOLOv8 detection bbox labels (`class cx cy w h`, normalized, class-agnostic single class
"tooth") for reproducing Ghorbani et al.'s Stage 1 detector (YOLOv8n). CHAI's own
YOLO/data.yaml already trains a class-agnostic YOLOv8n-SEG model from a pre-existing polygon
label set (dataset/{split}/labels/{jaw}/*.txt) - this script targets the same GT and the same
single "tooth" class, but derives a tight axis-aligned bounding box (min/max of each tooth's
segmentation polygon, matching ResNet/tooth/train.py's crop convention minus its 10px padding -
detection GT should reflect the tooth's real extent, not a crop margin) rather than a polygon,
since Ghorbani's Stage 1 is a plain detector (YOLOv8n), not a segmenter.

Images are not duplicated - dataset/{split}/images is 60+ GB per jaw/split alone, so this script
creates NTFS directory junctions (comparison/ghorbani/detect_dataset/{split}/images/{jaw} -> the
real dataset/{split}/images/{jaw}) instead of copying, and only writes the new label .txt files
for real. Ultralytics resolves an image's label path by swapping the last "images" path segment
for "labels" - since the junctioned images dir has the exact same {split}/images/{jaw} structure
as this script's own labels dir, that resolution lands on the label files this script writes,
not on CHAI's own YOLOv8n-seg labels living at dataset/{split}/labels/{jaw}.

Usage:
    .venv/Scripts/python.exe comparison/ghorbani/prepare_detect_labels.py
"""
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

GHORBANI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GHORBANI_DIR.parent.parent           # .../CHAI/CHAI
DATASET_DIR = PROJECT_ROOT.parent / "dataset"       # .../CHAI/dataset
OUT_DIR = GHORBANI_DIR / "detect_dataset"

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".webp"]


def junction(link_path, target_path):
    """Creates an NTFS directory junction (no admin rights needed, unlike symlinks) so the
    detection dataset can reuse CHAI's existing images without copying 60+ GB per jaw/split."""
    link_path = Path(link_path)
    if link_path.exists():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)],
                    check=True, capture_output=True)


def poly_to_bbox(seg):
    if isinstance(seg[0], list) and len(seg[0]) == 2:
        poly = np.array(seg, dtype=float)
    else:
        poly = np.array(seg, dtype=float).reshape(-1, 2)
    return poly.min(axis=0), poly.max(axis=0)


def process_split_jaw(split, jaw):
    json_dir = DATASET_DIR / split / "labels_json" / jaw
    image_dir = DATASET_DIR / split / "images" / jaw
    labels_out = OUT_DIR / split / "labels" / jaw
    labels_out.mkdir(parents=True, exist_ok=True)

    if image_dir.exists():
        junction(OUT_DIR / split / "images" / jaw, image_dir)

    if not json_dir.exists():
        print(f"Skip {split}/{jaw}: {json_dir} not found.")
        return 0, 0

    n_images, n_boxes = 0, 0
    for json_path in sorted(json_dir.glob("*.json")):
        stem = json_path.stem
        image_name = None
        for ext in IMAGE_EXTS:
            if (image_dir / f"{stem}{ext}").exists():
                image_name = f"{stem}{ext}"
                break
        if image_name is None:
            continue

        with Image.open(image_dir / image_name) as img:
            w, h = img.size

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        lines = []
        for t in data.get("tooth", []):
            seg = t.get("segmentation", [])
            if not seg:
                continue
            (x_min, y_min), (x_max, y_max) = poly_to_bbox(seg)
            cx = ((x_min + x_max) / 2) / w
            cy = ((y_min + y_max) / 2) / h
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            continue
        with open(labels_out / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        n_images += 1
        n_boxes += len(lines)

    return n_images, n_boxes


def main():
    data_yaml = OUT_DIR / "data.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(
            # Absolute, not "." - ultralytics resolves a relative `path:` against the CWD the
            # script is run from, not the yaml file's own directory, so "." broke as soon as
            # this was invoked from the project root instead of from detect_dataset/ itself.
            f"path: {OUT_DIR.as_posix()}\n"
            "train:\n  - train/images/lower\n  - train/images/upper\n"
            "val:\n  - val/images/lower\n  - val/images/upper\n"
            "test:\n  - test/images/lower\n  - test/images/upper\n"
            "nc: 1\n"
            "names: ['tooth']\n"
        )

    for split in ("train", "val", "test"):
        for jaw in ("lower", "upper"):
            n_images, n_boxes = process_split_jaw(split, jaw)
            print(f"{split}/{jaw}: {n_images} images, {n_boxes} boxes.")

    print(f"\nDone. data.yaml written to {data_yaml.resolve()}")


if __name__ == "__main__":
    main()
