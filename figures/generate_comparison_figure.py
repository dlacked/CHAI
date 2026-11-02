"""
Regenerates access.tex's Figure 4 (comparison.png) directly from already-computed per-model
predictions (chai_raw_vs_hungarian_per_jaw.pkl's "hungarian" entry, ghorbani_per_jaw.pkl,
yoon_per_jaw.pkl) and each image's ground-truth tooth polygons - no live model inference needed,
since every prediction is already keyed by (jaw, image_name, true FDI number) exactly as
js/render.js's own browser-based tool would show it (same green/red correctness convention,
same ground-truth-conditioned matching this study's evaluation uses throughout).

3 rows (Proposed / Ghorbani / Yoon) x 5 panels (a)-(e), each tooth boxed at its ground-truth
location with a colored FDI-number badge (green: correct, red: incorrect).

Usage:
    .venv/Scripts/python.exe figures/generate_comparison_figure.py
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

PANELS = [
    ("a", "upper", "upper_31193.png"),
    ("b", "lower", "lower_39956.png"),
    ("c", "lower", "lower_43012.png"),
    ("d", "upper", "upper_29579.png"),
    ("e", "upper", "upper_32003.png"),
]

GREEN = (27, 158, 63)
RED = (214, 39, 39)
WHITE = (255, 255, 255)
PANEL_LABEL_BG = (0, 0, 0)

ROW_LABELS = ["Proposed (Ours)", "Z. Ghorbani et al.", "K. Yoon et al."]


def load_gt_boxes(jaw, image_name):
    """{fdi_number: (x1,y1,x2,y2)} from the polygon's own min/max, no padding - just for box/
    badge placement, not a crop."""
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
    # {(jaw, image_name): {true_fdi: pred_fdi}}
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


def draw_badge(draw, x, y, text, color, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 10, 6
    box = (x, y, x + tw + pad_x * 2, y + th + pad_y * 2)
    draw.rounded_rectangle(box, radius=8, fill=color, outline=WHITE, width=2)
    draw.text((x + pad_x - bbox[0], y + pad_y - bbox[1]), text, fill=WHITE, font=font)
    return box


def render_panel(jaw, image_name, true_boxes, pred_maps):
    """Returns a list of 3 PIL Images (one per model row), each the full arch image with
    GT-conditioned predicted FDI badges, colored by correctness."""
    img_path = IMAGE_DIR / jaw / image_name
    base = Image.open(img_path).convert("RGB")
    font = get_font(max(18, base.width // 40))

    rows = []
    for pred_map in pred_maps:
        im = base.copy()
        draw = ImageDraw.Draw(im)
        for true_fdi, (x1, y1, x2, y2) in true_boxes.items():
            pred_fdi = pred_map.get(true_fdi)
            if pred_fdi is None:
                continue
            color = GREEN if pred_fdi == true_fdi else RED
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            draw_badge(draw, cx - 20, cy - 14, str(pred_fdi), color, font)
        rows.append(im)
    return rows


def main():
    chai_preds = load_predictions(
        ROOT / "ResNet" / "results" / "chai_raw_vs_hungarian_per_jaw.pkl", "hungarian")
    ghorbani_preds = load_predictions(ROOT / "comparison" / "ghorbani" / "results" / "ghorbani_per_jaw.pkl")
    yoon_preds = load_predictions(ROOT / "comparison" / "yoon" / "results" / "yoon_per_jaw.pkl")

    panel_images = []  # list of (label, [row0_img, row1_img, row2_img])
    for label, jaw, image_name in PANELS:
        true_boxes = load_gt_boxes(jaw, image_name)
        key = (jaw, image_name)
        pred_maps = [chai_preds.get(key, {}), ghorbani_preds.get(key, {}), yoon_preds.get(key, {})]
        rows = render_panel(jaw, image_name, true_boxes, pred_maps)
        panel_images.append((label, rows))
        print(f"Rendered panel ({label}): {image_name}")

    # --- Assemble 3 rows x 5 columns grid, scaling each panel's column to a common cell size ---
    cell_w = 420
    cell_h = 480
    label_h = 40
    row_label_w = 200
    margin = 12

    grid_w = row_label_w + len(panel_images) * (cell_w + margin) + margin
    grid_h = label_h + 3 * (cell_h + margin) + margin
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    font_label = get_font(22)
    font_row = get_font(20)

    for row_idx, row_label in enumerate(ROW_LABELS):
        ry = label_h + margin + row_idx * (cell_h + margin) + cell_h // 2
        draw.text((margin, ry), row_label, fill=(0, 0, 0), font=font_row)

    for col_idx, (label, rows) in enumerate(panel_images):
        cx = row_label_w + margin + col_idx * (cell_w + margin)
        draw.text((cx + cell_w // 2 - 8, 6), f"({label})", fill=(0, 0, 0), font=font_label)
        for row_idx, im in enumerate(rows):
            cy = label_h + margin + row_idx * (cell_h + margin)
            thumb = im.copy()
            thumb.thumbnail((cell_w, cell_h))
            paste_x = cx + (cell_w - thumb.width) // 2
            paste_y = cy + (cell_h - thumb.height) // 2
            grid.paste(thumb, (paste_x, paste_y))

    out_path = ROOT / "results_comparison_figure.png"
    grid.save(out_path, dpi=(200, 200))
    print(f"\nSaved assembled figure to {out_path}")


if __name__ == "__main__":
    main()
