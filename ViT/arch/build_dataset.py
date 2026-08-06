import os
import json
import argparse
import importlib.util
from pathlib import Path
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms

ARCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARCH_DIR.parent.parent
RESNET_TOOTH_TRAIN_PATH = PROJECT_ROOT / "ResNet" / "tooth" / "train.py"

MAX_TEETH_PER_ARCH = 12  # matches the 12 FDI slots (1-6) used per side/jaw everywhere else
DEFAULT_CROP_WORKERS = 6  # kept low regardless of core count - this machine is RAM-bound, not CPU-bound
DEFAULT_BATCH_TEETH = 96  # teeth per ResNet forward call, ~batch_teeth/12 arches at a time


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Reuse ResNet/tooth/train.py's crop logic and model architecture instead of reimplementing
# them - loaded by path (not `sys.path` + `import train`) to avoid clashing with this
# directory's own train.py module name.
_resnet_tooth_train = _load_module("resnet_tooth_train", str(RESNET_TOOTH_TRAIN_PATH))
ToothDataset = _resnet_tooth_train.ToothDataset
ToothPositionClassifier = _resnet_tooth_train.ToothPositionClassifier

# Same val-time transform as ResNet/tooth/train.py / server.py (no augmentation - this cache
# is reused across every ViT training epoch).
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def crop_teeth_for_image(dataset_dir, split, jaw, image_name, rows, transform):
    """Crops every tooth belonging to one arch image in a single pass. Mirrors
    ResNet/tooth/train.py's ToothDataset.__getitem__ crop logic exactly, but reads the source
    image and its GT JSON once per arch instead of once per tooth (ToothDataset re-reads both
    on every __getitem__ call, which is fine for shuffled single-tooth training batches but
    makes building a full arch sequence ~12x slower than necessary).
    """
    img_path = Path(dataset_dir) / split / "images" / jaw / image_name
    image = cv2.imread(str(img_path))
    if image is None:
        image = np.zeros((224, 224, 3), dtype=np.uint8)
    h, w = image.shape[:2]

    image_name_no_ext = os.path.splitext(image_name)[0]
    json_path = Path(dataset_dir) / split / "labels_json" / jaw / f"{image_name_no_ext}.json"
    fdi_to_poly = {}
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for t in data.get("tooth", []):
                num = t.get("teeth_num")
                seg = t.get("segmentation", [])
                if num is not None and seg:
                    fdi_to_poly[num] = seg
        except Exception as e:
            print(f"Error loading GT json for crop ({json_path}): {e}")

    imgs, metas, labels = [], [], []
    for row in rows:
        fdi_number = int(row['fdi_number'])
        target_label = int(row['fdi_last_digit']) - 1

        x1, y1, x2, y2 = 0, 0, w, h
        seg = fdi_to_poly.get(fdi_number)
        if seg:
            if isinstance(seg[0], list) and len(seg[0]) == 2:
                poly = np.array(seg)
            else:
                poly = np.array(seg).reshape(-1, 2)
            x_min, y_min = poly.min(axis=0)
            x_max, y_max = poly.max(axis=0)
            pad = 10
            x1 = int(max(0, x_min - pad))
            y1 = int(max(0, y_min - pad))
            x2 = int(min(w, x_max + pad))
            y2 = int(min(h, y_max + pad))

        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            cropped = np.zeros((224, 224, 3), dtype=np.uint8)
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        img_tensor = transform(Image.fromarray(cropped_rgb))

        meta = torch.tensor([
            float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2']), float(row['theta'])
        ], dtype=torch.float32)

        imgs.append(img_tensor)
        metas.append(meta)
        labels.append(target_label)

    return imgs, metas, labels


def load_resnet(jaw, model_dir, device):
    weights_path = Path(model_dir) / f"{jaw}_best.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"ResNet weights not found: {weights_path}")
    model = ToothPositionClassifier(num_classes=6, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def build_split(jaw, split, dataset_dir, csv_dir, model_dir, device,
                 crop_workers=DEFAULT_CROP_WORKERS, batch_teeth=DEFAULT_BATCH_TEETH):
    csv_path = Path(csv_dir) / f"{jaw}_features_{split}.csv"
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}. Skipping {jaw}/{split}.")
        return []

    dataset = ToothDataset(csv_path=csv_path, dataset_dir=dataset_dir, jaw=jaw, split=split, transform=val_transform)
    if len(dataset) == 0:
        print(f"No rows for {jaw}/{split}, skipping.")
        return []

    resnet = load_resnet(jaw, model_dir, device)

    # CSV rows are written image-by-image, arch-ordered left to right (functions/features/main.py),
    # and ToothDataset.rows preserves that order - so consecutive rows sharing image_name are one arch.
    groups = OrderedDict()
    for row in dataset.rows:
        groups.setdefault(row['image_name'], []).append(row)

    items = []
    skipped_long = 0
    for image_name, rows in groups.items():
        if len(rows) > MAX_TEETH_PER_ARCH:
            skipped_long += 1
            rows = rows[:MAX_TEETH_PER_ARCH]
        items.append((image_name, rows))

    def crop_job(image_name, rows):
        return crop_teeth_for_image(dataset_dir, split, jaw, image_name, rows, val_transform)

    def flush(pending):
        # One ResNet forward call per batch of arches instead of per arch (<=12 teeth) - the
        # original per-arch batch size left the GPU underused relative to the CPU crop/decode cost.
        all_imgs = torch.cat([torch.stack(imgs) for _, imgs, _, _ in pending]).to(device)
        all_metas = torch.cat([torch.stack(metas) for _, _, metas, _ in pending]).to(device)
        with torch.no_grad():
            img_vec = resnet.resnet(all_imgs)                                    # (N, 512)
            meta_feat = resnet.meta_fc(all_metas)                                # (N, 256)
            logits = resnet.classifier(torch.cat([img_vec, meta_feat], dim=1))
            prob_vec = torch.softmax(logits, dim=1)                              # (N, 6)

        out = []
        offset = 0
        for image_name, imgs, _, labels in pending:
            k = len(imgs)
            out.append({
                "image_name": image_name,
                "geom": all_metas[offset:offset + k].cpu(),
                "img_vec": img_vec[offset:offset + k].cpu(),
                "prob_vec": prob_vec[offset:offset + k].cpu(),
                "target": torch.tensor(labels, dtype=torch.long),
            })
            offset += k
        return out

    # Crop/decode runs on a bounded thread pool (cv2/PIL/torch ops release the GIL, so this
    # overlaps disk + CPU crop work with the GPU forward pass) while keeping only `window` arches'
    # worth of cropped tensors resident at once, instead of buffering the whole split in memory.
    sequences = []
    pending, pending_teeth = [], 0
    window = crop_workers * 2  # kept small - each slot holds a full decoded page image + crops
    with ThreadPoolExecutor(max_workers=crop_workers) as executor:
        it = iter(items)
        futures = deque()

        def submit_next():
            image_name, rows = next(it, (None, None))
            if image_name is None:
                return False
            futures.append((image_name, executor.submit(crop_job, image_name, rows)))
            return True

        for _ in range(window):
            if not submit_next():
                break

        while futures:
            image_name, fut = futures.popleft()
            imgs, metas, labels = fut.result()
            submit_next()

            n = len(imgs)
            if pending and pending_teeth + n > batch_teeth:
                sequences.extend(flush(pending))
                pending, pending_teeth = [], 0
            pending.append((image_name, imgs, metas, labels))
            pending_teeth += n

    if pending:
        sequences.extend(flush(pending))

    if skipped_long:
        print(f"Warning: {skipped_long} arches in {jaw}/{split} had more than {MAX_TEETH_PER_ARCH} teeth; truncated.")

    return sequences


def main():
    dataset_dir_default = PROJECT_ROOT.parent / "dataset"
    csv_dir_default = PROJECT_ROOT / "ResNet" / "tooth" / "csv"
    model_dir_default = PROJECT_ROOT / "ResNet" / "tooth" / "model"
    cache_dir_default = ARCH_DIR / "cache"

    parser = argparse.ArgumentParser(description="Build offline arch-sequence cache for the ViT tooth-number refiner.")
    parser.add_argument("--dataset_dir", type=str, default=str(dataset_dir_default))
    parser.add_argument("--csv_dir", type=str, default=str(csv_dir_default))
    parser.add_argument("--model_dir", type=str, default=str(model_dir_default))
    parser.add_argument("--cache_dir", type=str, default=str(cache_dir_default))
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache files")
    parser.add_argument("--crop_workers", type=int, default=DEFAULT_CROP_WORKERS,
                         help="Threads used to decode/crop images concurrently with GPU forward passes")
    parser.add_argument("--batch_teeth", type=int, default=DEFAULT_BATCH_TEETH,
                         help="Teeth per ResNet forward call (multiple arches batched together)")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for jaw in ("lower", "upper"):
        for split in ("train", "val"):
            out_path = cache_dir / f"{jaw}_{split}.pt"
            if out_path.exists() and not args.force:
                print(f"Cache already exists: {out_path}. Skipping (use --force to overwrite).")
                continue
            print(f"\nBuilding {jaw}/{split} arch sequences...")
            sequences = build_split(jaw, split, args.dataset_dir, args.csv_dir, args.model_dir, device,
                                     crop_workers=args.crop_workers, batch_teeth=args.batch_teeth)
            if not sequences:
                continue
            n_teeth = sum(len(s["target"]) for s in sequences)
            print(f"  {len(sequences)} arches, {n_teeth} teeth.")
            torch.save(sequences, out_path)
            print(f"  Saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
