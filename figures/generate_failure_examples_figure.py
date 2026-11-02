"""
Regenerates access.tex's Figure 5 (comp3_chai_failure_examples.png) from the proposed model's
RAW (pre-Hungarian) per-tooth predictions in chai_raw_vs_hungarian_per_jaw.pkl.

Style: a tight crop around just the teeth involved, thick black/red tooth boxes, and a white
label box anchored at each tooth box's own top-left corner ("pred=N" in black when correct,
"pred=N" / "GT=N" in red on two lines when wrong) - no leader lines, no floating placement.
Font size and line width are each computed per panel so that, after both panels are thumbnailed
to the same display height, the FINAL on-page label size and border thickness are identical
across panels regardless of each panel's own crop resolution (this is what made panel (a)'s
labels look smaller than panel (b)'s in the previous version - same source font size drawn into
two differently-sized crops, then both thumbnailed to one common height).

Panel (a): a chain-shift error (upper_31380.png) - FDI13 confidently predicts FDI14's class and
FDI14 confidently predicts FDI15's class (both ~100% confidence), while FDI15 is itself missing
from this arch, so neither predicted class collides with any other tooth's claim and Hungarian
assignment cannot catch it. (Re-derived this session: a search across the full test set for a
genuine A<->B mutual swap, matching the figure's original wording, found zero such cases - FDI22
in particular is never misclassified anywhere in the current evaluation run - so this chain-shift
case, verified against the same raw/hungarian pickle, replaces it as the real illustrative example
of the same underlying point: a confident error that doesn't violate the capacity-1 constraint.)

Panel (b): a single-tooth error (upper_30079.png) - FDI11 misclassified as FDI12 at 0.7454
confidence, matching the figure's original 74.5% figure exactly.

Usage:
    .venv/Scripts/python.exe figures/generate_failure_examples_figure.py
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

BLACK = (0, 0, 0)
RED = (214, 39, 39)
WHITE = (255, 255, 255)

CELL_H = 420  # common on-page display height both panels are thumbnailed to
TARGET_FONT_MAIN = 40   # desired FINAL (post-thumbnail) px size of the "pred=N" line
TARGET_FONT_SMALL = 33  # desired FINAL px size of the "GT=N" line
TARGET_BOX_WIDTH = 3    # desired FINAL px tooth-box border thickness

# (panel label, jaw, image file, [FDI numbers to show], crop margin L/R/T/B)
PANELS = [
    ("a", "upper", "upper_31380.png", [12, 13, 14], (90, 110, 230, 40)),
    ("b", "upper", "upper_30079.png", [11, 12, 13, 21], (90, 90, 230, 40)),
]


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


def load_raw_predictions(pkl_path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    raw = d["raw"]
    out = {}
    for jaw in ("lower", "upper"):
        dd = raw[jaw]
        for pred_digit, tens_pred, fdi_number, image_name in zip(
                dd["y_pred"], dd["y_tens_pred"], dd["y_fdi_number"], dd["y_image_name"]):
            key = (jaw, image_name)
            pred_fdi = tens_pred * 10 + (pred_digit + 1)
            out.setdefault(key, {})[fdi_number] = pred_fdi
    return out


def get_font(size, bold=True):
    names = ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf") if bold else \
            ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def compute_crop(image_path, fdi_list, true_boxes, margins):
    with Image.open(image_path) as im:
        w, h = im.size
    xs1, ys1, xs2, ys2 = [], [], [], []
    for fdi in fdi_list:
        if fdi in true_boxes:
            x1, y1, x2, y2 = true_boxes[fdi]
            xs1.append(x1); ys1.append(y1); xs2.append(x2); ys2.append(y2)
    left_m, right_m, top_m, bottom_m = margins
    crop_x1 = max(0, min(xs1) - left_m)
    crop_y1 = max(0, min(ys1) - top_m)
    crop_x2 = min(w, max(xs2) + right_m)
    crop_y2 = min(h, max(ys2) + bottom_m)
    return crop_x1, crop_y1, crop_x2, crop_y2


def draw_tooth_box(draw, tooth_box, is_correct, box_width):
    color = BLACK if is_correct else RED
    draw.rectangle(tooth_box, outline=color, width=box_width)


def draw_tooth_label(draw, tooth_box, pred_fdi, true_fdi, is_correct, font_main, font_small, top_bound):
    # Drawn in a separate pass, after every tooth box, so a label's opaque white background
    # always sits on top of any box border beneath it - borders never cross through label text.
    color = BLACK if is_correct else RED
    lines = [f"pred={pred_fdi}"] if is_correct else [f"pred={pred_fdi}", f"GT={true_fdi}"]
    fonts = [font_main] if is_correct else [font_main, font_small]

    widths, heights, bboxes = [], [], []
    for line, f in zip(lines, fonts):
        bbox = draw.textbbox((0, 0), line, font=f)
        bboxes.append(bbox)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])

    pad_x, pad_y, line_gap = 10, 6, 4
    box_w = max(widths) + pad_x * 2
    box_h = sum(heights) + pad_y * 2 + line_gap * (len(lines) - 1)

    tx1, ty1, tx2, ty2 = tooth_box
    # Label anchored flush with the tooth box's own top-left corner, sitting just above it -
    # unless that would push it above the image's own top edge, in which case it flips to sit
    # just below the box's bottom-left corner instead (e.g. a tooth near the gumline/image edge).
    if ty1 - box_h < top_bound:
        lb = (tx1, ty2, tx1 + box_w, ty2 + box_h)
    else:
        lb = (tx1, ty1 - box_h, tx1 + box_w, ty1)

    draw.rectangle(lb, fill=WHITE)
    y = lb[1] + pad_y
    for line, f, bbox, w, h in zip(lines, fonts, bboxes, widths, heights):
        draw.text((tx1 + pad_x - bbox[0], y - bbox[1]), line, fill=color, font=f)
        y += h + line_gap


def render_panel(jaw, image_name, fdi_list, pred_map, true_boxes, margins):
    img_path = IMAGE_DIR / jaw / image_name
    base = Image.open(img_path).convert("RGB")

    crop_x1, crop_y1, crop_x2, crop_y2 = compute_crop(img_path, fdi_list, true_boxes, margins)
    crop_h = crop_y2 - crop_y1
    scale = CELL_H / crop_h  # how much this crop will be shrunk when thumbnailed to CELL_H

    font_main = get_font(round(TARGET_FONT_MAIN / scale))
    font_small = get_font(round(TARGET_FONT_SMALL / scale))
    box_width = max(1, round(TARGET_BOX_WIDTH / scale))

    draw = ImageDraw.Draw(base)
    ordered = sorted((fdi for fdi in fdi_list if fdi in true_boxes),
                      key=lambda f: (true_boxes[f][0] + true_boxes[f][2]) / 2)

    # Pass 1: every tooth box, so none of their borders can later cross over a label (pass 2).
    for fdi in ordered:
        pred_fdi = pred_map.get(fdi, fdi)
        draw_tooth_box(draw, true_boxes[fdi], pred_fdi == fdi, box_width)

    # Pass 2: every label, drawn last so each one's opaque background sits on top of all boxes.
    for fdi in ordered:
        tooth_box = true_boxes[fdi]
        pred_fdi = pred_map.get(fdi, fdi)
        is_correct = pred_fdi == fdi
        draw_tooth_label(draw, tooth_box, pred_fdi, fdi, is_correct, font_main, font_small, top_bound=crop_y1)

    return base.crop((int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)))


def main():
    chai_preds = load_raw_predictions(ROOT / "ResNet" / "results" / "chai_raw_vs_hungarian_per_jaw.pkl")

    panel_images = []
    for label, jaw, image_name, fdi_list, margins in PANELS:
        true_boxes = load_gt_boxes(jaw, image_name)
        pred_map = chai_preds.get((jaw, image_name), {})
        im = render_panel(jaw, image_name, fdi_list, pred_map, true_boxes, margins)
        panel_images.append((label, im))
        print(f"Rendered panel ({label}): {image_name}, size={im.size}")

    label_h = 44
    margin = 16

    scaled = []
    for label, im in panel_images:
        thumb = im.copy()
        thumb.thumbnail((10000, CELL_H))
        scaled.append((label, thumb))

    grid_w = sum(t.width for _, t in scaled) + margin * (len(scaled) + 1)
    grid_h = label_h + CELL_H + margin * 2
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    font_label = get_font(26)

    x = margin
    for label, thumb in scaled:
        draw.text((x + thumb.width // 2 - 8, 6), f"({label})", fill=(0, 0, 0), font=font_label)
        grid.paste(thumb, (x, label_h + margin))
        x += thumb.width + margin

    out_path = ROOT / "results_failure_examples_figure.png"
    grid.save(out_path, dpi=(200, 200))
    print(f"\nSaved assembled figure to {out_path}")


if __name__ == "__main__":
    main()
