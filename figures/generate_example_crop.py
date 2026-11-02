"""
One-off: a tight crop of lower_42019.png around FDI31/FDI32 with large rounded FDI-number
badges (matching Figure 4's drawFdiNumberBadge style: green=correct, red=incorrect, white
border), at a big, paper-ready font size.

Usage:
    .venv/Scripts/python.exe figures/generate_example_crop.py
"""
import json
import pickle
import sys
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

GREEN = (27, 158, 63)
RED = (214, 39, 39)
WHITE = (255, 255, 255)

JAW = "lower"
IMAGE_NAME = "lower_42019.png"
FDI_LIST = [31, 32]
MARGIN = (120, 120, 100, 100)  # left, right, top, bottom
FONT_SIZE = 90


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


def load_predictions(pkl_path, hungarian_key=None):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    if hungarian_key:
        d = d[hungarian_key]
    out = {}
    for jaw in ("lower", "upper"):
        dd = d[jaw]
        for pred_digit, tens_pred, fdi_number, image_name in zip(
                dd["y_pred"], dd["y_tens_pred"], dd["y_fdi_number"], dd["y_image_name"]):
            key = (jaw, image_name)
            pred_fdi = tens_pred * 10 + (pred_digit + 1)
            out.setdefault(key, {})[fdi_number] = pred_fdi
    return out


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
    pad_x, pad_y = 22, 14
    box_w, box_h = total_w + pad_x * 2, nh + pad_y * 2
    box = (cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2)
    draw.rounded_rectangle(box, radius=16, fill=color, outline=WHITE, width=4)

    start_x = cx - total_w / 2
    draw.text((start_x - prefix_bbox[0], cy - ph / 2 - prefix_bbox[1]), prefix, fill=WHITE, font=font_prefix)
    draw.text((start_x + pw + spacing - num_bbox[0], cy - nh / 2 - num_bbox[1]), num_text, fill=WHITE, font=font_num)


def main():
    chai_preds = load_predictions(
        ROOT / "ResNet" / "results" / "chai_raw_vs_hungarian_per_jaw.pkl", "hungarian")
    true_boxes = load_gt_boxes(JAW, IMAGE_NAME)
    pred_map = chai_preds.get((JAW, IMAGE_NAME), {})

    img_path = IMAGE_DIR / JAW / IMAGE_NAME
    base = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(base)
    font_num = get_font(FONT_SIZE)
    font_prefix = get_font(round(FONT_SIZE * 0.4))

    xs1, ys1, xs2, ys2 = [], [], [], []
    for fdi in FDI_LIST:
        x1, y1, x2, y2 = true_boxes[fdi]
        xs1.append(x1); ys1.append(y1); xs2.append(x2); ys2.append(y2)
        pred_fdi = pred_map.get(fdi, fdi)
        color = GREEN if pred_fdi == fdi else RED
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        draw_badge(draw, cx, cy, str(pred_fdi), color, font_num, font_prefix)

    left_m, right_m, top_m, bottom_m = MARGIN
    crop = base.crop((
        max(0, int(min(xs1) - left_m)),
        max(0, int(min(ys1) - top_m)),
        min(base.width, int(max(xs2) + right_m)),
        min(base.height, int(max(ys2) + bottom_m)),
    ))

    out_path = ROOT / "results_example_crop_42019.png"
    crop.save(out_path, dpi=(200, 200))
    print(f"Saved to {out_path}, size={crop.size}")


if __name__ == "__main__":
    main()
