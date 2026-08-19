"""
Converts CHAI's per-tooth GT polygon labels (dataset/{split}/labels_json/{jaw}/*.json) into a
single COCO-format detection annotation file per split, for reproducing Yoon et al.'s tooth-
number-recognition model (a Cascade R-CNN, trained via mmdetection - see train.py/config.py in
this folder).

Scope: Yoon et al.'s own model jointly detects tooth numbers AND stages dental caries in full
intraoral images (occlusal + frontal + lateral views). CHAI's dataset has neither caries labels
nor frontal/lateral views, so this reproduction covers tooth-number recognition only, on CHAI's
own occlusal upper/lower images - the same "same-dataset, same task" adaptation already applied
to the Ghorbani reproduction (see comparison/ghorbani/prepare_detect_labels.py's docstring for
the identical reasoning). Both jaws are pooled into one COCO dataset per split (mirroring
Ghorbani's classify stage) since Yoon's own model isn't jaw-segregated either - it detects
whatever tooth numbers appear in a given intraoral image regardless of view.

Classes: the 24 full two-digit FDI numbers this dataset actually has GT for (11-16/21-26/31-36/
41-46 - no primary teeth, no 7/8 wisdom teeth, same as Ghorbani's reproduction and for the same
reason: CHAI's own dataset doesn't contain them).

Images are not duplicated - dataset/{split}/images is 60+ GB per jaw/split alone, so this script
creates NTFS directory junctions (comparison/yoon/coco_dataset/{split}/images/{jaw} -> the real
dataset/{split}/images/{jaw}) instead of copying, matching prepare_detect_labels.py's approach.

Usage:
    .venv/Scripts/python.exe comparison/yoon/prepare_coco_labels.py
"""
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

YOON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = YOON_DIR.parent.parent           # .../CHAI/CHAI
DATASET_DIR = PROJECT_ROOT.parent / "dataset"   # .../CHAI/dataset
OUT_DIR = YOON_DIR / "coco_dataset"

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".webp"]

# Full two-digit FDI numbers this dataset actually has GT for (digits 1-6 only, both jaws pooled -
# see this module's docstring). category_id in the COCO annotations below is the 0-based index
# into this list, matching the "classes" tuple order the mmdetection config will use.
FDI_CLASSES = [q * 10 + d for q in (1, 2, 3, 4) for d in range(1, 7)]
FDI_TO_IDX = {n: i for i, n in enumerate(FDI_CLASSES)}


def junction(link_path, target_path):
    """Creates an NTFS directory junction (no admin rights needed, unlike symlinks) so the COCO
    dataset can reuse CHAI's existing images without copying 60+ GB per jaw/split."""
    link_path = Path(link_path)
    if link_path.exists():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)],
                    check=True, capture_output=True)


def poly_bbox(seg):
    if isinstance(seg[0], list) and len(seg[0]) == 2:
        poly = np.array(seg, dtype=float)
    else:
        poly = np.array(seg, dtype=float).reshape(-1, 2)
    return poly.min(axis=0), poly.max(axis=0)


def build_split(split):
    images, annotations = [], []
    image_id, ann_id = 0, 0

    for jaw in ("lower", "upper"):
        json_dir = DATASET_DIR / split / "labels_json" / jaw
        image_dir = DATASET_DIR / split / "images" / jaw
        if not json_dir.exists():
            print(f"Skip {split}/{jaw}: {json_dir} not found.")
            continue

        junction(OUT_DIR / split / "images" / jaw, image_dir)

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

            boxes = []
            for t in data.get("tooth", []):
                fdi_number = t.get("teeth_num")
                seg = t.get("segmentation", [])
                if fdi_number not in FDI_TO_IDX or not seg:
                    continue
                (x_min, y_min), (x_max, y_max) = poly_bbox(seg)
                boxes.append((fdi_number, float(x_min), float(y_min),
                              float(x_max), float(y_max)))
            if not boxes:
                continue

            # file_name is relative to OUT_DIR / split / "images" - mmdetection resolves it
            # against the config's data_prefix, not this JSON's own location.
            images.append({
                "id": image_id,
                "file_name": f"{jaw}/{image_name}",
                "width": w,
                "height": h,
            })
            for fdi_number, x_min, y_min, x_max, y_max in boxes:
                bw, bh = x_max - x_min, y_max - y_min
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": FDI_TO_IDX[fdi_number],
                    "bbox": [x_min, y_min, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                })
                ann_id += 1
            image_id += 1

    categories = [{"id": i, "name": str(n)} for i, n in enumerate(FDI_CLASSES)]
    coco = {"images": images, "annotations": annotations, "categories": categories}

    out_path = OUT_DIR / split / "annotations.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    print(f"{split}: {len(images)} images, {len(annotations)} boxes -> {out_path}")
    return len(images), len(annotations)


def main():
    for split in ("train", "val", "test"):
        build_split(split)


if __name__ == "__main__":
    main()
