"""
No-rotation counterpart to tooth.py's compute_tens_predictions: replicates production's
pre-Hungarian FDI tens-digit (quadrant) guess, but decides left/right from the image's own
horizontal center (width/2) instead of a per-arch PCA axis (mean_pt + angle).

Same reasoning as functions/features/coords_baseline.py's X mirror: the capture protocol frames
the arch centered in the photo (verified: complete arches average ~2deg of PCA-estimated tilt,
92% within 5deg), and a single missing posterior molar swings the PCA axis by ~13deg on average
(leave-one-out test, see project notes) - which is exactly the mechanism behind the documented
"a real 41 and 31 both reading as 31" bug in compute_tens_predictions's docstring. Using width/2
instead removes that dependency on which teeth happen to be present entirely.

STATUS (2026-08-31): originally built as a comparison-only script (PCA won by ~0.03-0.04pp on a
held-out missing-teeth test, see project notes), but the user chose width/2 anyway for pipeline
consistency (no PCA left anywhere in the tooth-numbering post-process). This is now what
production actually does - js/geometry.js's computeToothMetaPooled/computeQuadrantTens and
functions/graph/tooth.py's evaluate_tooth_number both call this (or its JS equivalent) instead of
the PCA-based compute_tens_predictions, which is kept only as the ablation comparison point.

Returns the same {(image_name, fdi_number): predicted_tens} shape as compute_tens_predictions.
"""
import json
from pathlib import Path

from PIL import Image


def compute_tens_predictions_norot(dataset_dir, jaw, split):
    image_exts = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".webp"]
    json_dir = Path(dataset_dir) / split / "labels_json" / jaw
    image_dir = Path(dataset_dir) / split / "images" / jaw

    result = {}
    for json_path in sorted(json_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        segs = []
        for t in data.get("tooth", []):
            num = t.get("teeth_num")
            seg = t.get("segmentation", [])
            if num is None or not seg:
                continue
            poly = seg if (isinstance(seg[0], list) and len(seg[0]) == 2) else \
                [[seg[i], seg[i + 1]] for i in range(0, len(seg), 2)]
            cx = sum(p[0] for p in poly) / len(poly)
            segs.append((num, cx))
        if not segs:
            continue

        image_name = None
        img_path = None
        for ext in image_exts:
            cand = image_dir / f"{json_path.stem}{ext}"
            if cand.exists():
                image_name = f"{json_path.stem}{ext}"
                img_path = cand
                break
        if image_name is None:
            continue

        with Image.open(img_path) as im:
            width = im.size[0]

        for num, cx in segs:
            mirrored = cx >= width / 2
            if jaw == "upper":
                tens = 2 if mirrored else 1
            else:
                tens = 3 if mirrored else 4
            result[(image_name, num)] = tens

    return result
