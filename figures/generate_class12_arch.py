"""
Full-arch view of lower_42019.png with each tooth's RAW (pre-Hungarian) predicted local
class12 (0-11) as a large badge - the capacity-1 collision source for Figure 3
(comp3_hungarian_fix_example.png): FDI31 and FDI32 both predict class 0.

Badge color: side A (local quadrant 0, FDI 3x) = orange, side B (local quadrant 1, FDI 4x) =
gray, any class value that's duplicated in this arch (the actual collision) = red, overriding
the side color.

Usage:
    .venv/Scripts/python.exe figures/generate_class12_arch.py
"""
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_bootstrap_root = Path(__file__).resolve().parent.parent
if str(_bootstrap_root) not in sys.path:
    sys.path.append(str(_bootstrap_root))
from config import PathConfig
ROOT = PathConfig.ROOT
DATASET_DIR = ROOT.parent / "dataset"
IMAGE_DIR = DATASET_DIR / "test" / "images"
JSON_DIR = DATASET_DIR / "test" / "labels_json"

JAW = "lower"
IMAGE_NAME = "lower_42019.png"
FONT_SIZE = 100

GRAY = (90, 98, 110)
ORANGE = GRAY
RED = (214, 39, 39)
WHITE = (255, 255, 255)

JAW_TENS = {"lower": (3, 4), "upper": (1, 2)}


def load_gt_boxes(jaw, image_name):
    stem = Path(image_name).stem
    with open(JSON_DIR / jaw / f"{stem}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    boxes = {}
    for t in data.get("tooth", []):
        num = t.get("teeth_num")
        seg = t.get("segmentation", [])
        if num is None or not seg:
            continue
        xs = [p[0] for p in seg]
        ys = [p[1] for p in seg]
        boxes[num] = (min(xs), min(ys), max(xs), max(ys))
    return boxes


def get_font(size):
    for name in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_badge(draw, cx, cy, num_text, color, font_num, font_prefix, prefix="#"):
    prefix_bbox = draw.textbbox((0, 0), prefix, font=font_prefix)
    pw, ph = prefix_bbox[2] - prefix_bbox[0], prefix_bbox[3] - prefix_bbox[1]
    num_bbox = draw.textbbox((0, 0), num_text, font=font_num)
    nw, nh = num_bbox[2] - num_bbox[0], num_bbox[3] - num_bbox[1]

    spacing = 6
    total_w = pw + spacing + nw
    pad_x, pad_y = 18, 12
    box_w, box_h = total_w + pad_x * 2, nh + pad_y * 2
    box = (cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2)
    draw.rounded_rectangle(box, radius=14, fill=color, outline=WHITE, width=4)

    start_x = cx - total_w / 2
    draw.text((start_x - prefix_bbox[0], cy - ph / 2 - prefix_bbox[1]), prefix, fill=WHITE, font=font_prefix)
    draw.text((start_x + pw + spacing - num_bbox[0], cy - nh / 2 - num_bbox[1]), num_text, fill=WHITE, font=font_num)


def main():
    with open(ROOT / "ResNet" / "results" / "chai_raw_vs_hungarian_per_jaw.pkl", "rb") as f:
        d = pickle.load(f)
    raw = d["raw"][JAW]

    class12_by_fdi = {}
    for pred_digit, tens_pred, fdi, image_name in zip(
            raw["y_pred"], raw["y_tens_pred"], raw["y_fdi_number"], raw["y_image_name"]):
        if image_name != IMAGE_NAME:
            continue
        local_quadrant = 0 if tens_pred == JAW_TENS[JAW][0] else 1
        class12_by_fdi[fdi] = local_quadrant * 6 + pred_digit

    class_counts = Counter(class12_by_fdi.values())
    duplicated_classes = {c for c, n in class_counts.items() if n > 1}

    true_boxes = load_gt_boxes(JAW, IMAGE_NAME)
    img_path = IMAGE_DIR / JAW / IMAGE_NAME
    base = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(base)
    font_num = get_font(FONT_SIZE)
    font_prefix = get_font(round(FONT_SIZE * 0.4))

    for fdi, class12 in class12_by_fdi.items():
        x1, y1, x2, y2 = true_boxes[fdi]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if class12 in duplicated_classes:
            color = RED
        else:
            local_quadrant = 0 if fdi // 10 == JAW_TENS[JAW][0] else 1
            color = ORANGE if local_quadrant == 0 else GRAY
        draw_badge(draw, cx, cy, str(class12), color, font_num, font_prefix)

    out_path = ROOT / "results_class12_arch_42019.png"
    base.save(out_path, dpi=(200, 200))
    print(f"Saved to {out_path}, size={base.size}")
    print(f"Duplicated classes (collision): {duplicated_classes}")
    for fdi, c in sorted(class12_by_fdi.items()):
        print(f"  FDI{fdi}: class12={c}{'  <-- COLLISION' if c in duplicated_classes else ''}")


if __name__ == "__main__":
    main()
