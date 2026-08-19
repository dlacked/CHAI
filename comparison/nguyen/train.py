"""
Reproduces Nguyen et al.'s SegmentAnyTooth training setup: view-specific YOLO11n-seg models (they
trained 4, one per view - upper occlusal, lower occlusal, frontal, lateral; this dataset only has
occlusal-view images for the two jaws, so only those two are reproducible here) instead of CHAI's
own single upper+lower-combined YOLOv8n-seg model (see YOLO/train.py's docstring - "상·하악 통합
모델").

No label conversion is needed at all - dataset/{split}/labels/{jaw}/*.txt is already single-class
YOLO-seg polygon format (see YOLO/data.yaml), exactly what a per-jaw YOLO11n-seg model needs; this
script just points a jaw-scoped data.yaml straight at the existing dataset directories rather than
duplicating or converting anything.

Hyperparameters match the paper's stated setup exactly rather than CHAI's own YOLOv8n-seg training
config (imgsz 640, batch 8, patience 20 - see YOLO/train.py): imgsz 1024, batch 16, up to 2000
epochs with patience 300. The paper states ultralytics==8.3.18; this project has 8.4.91 installed
(newer) - not pinned down to match exactly, since downgrading globally risks breaking the rest of
the project's YOLO usage (server.py, YOLO/train.py) that's already validated against 8.4.91. Flag
this version gap if a discrepancy shows up against the paper's own numbers.

Usage:
    .venv/Scripts/python.exe comparison/nguyen/train.py --jaw upper
    .venv/Scripts/python.exe comparison/nguyen/train.py --jaw lower
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

NGUYEN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NGUYEN_DIR.parent.parent          # .../CHAI/CHAI
DATASET_DIR = PROJECT_ROOT.parent / "dataset"    # .../CHAI/dataset


def write_data_yaml(jaw):
    data_yaml = NGUYEN_DIR / f"data_{jaw}.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(
            f"path: {DATASET_DIR.as_posix()}\n"
            f"train: train/images/{jaw}\n"
            f"val: val/images/{jaw}\n"
            f"test: test/images/{jaw}\n"
            "nc: 1\n"
            "names: ['tooth']\n"
        )
    return data_yaml


def main():
    parser = argparse.ArgumentParser(description="Train a Nguyen-et-al.-style view-specific YOLO11n-seg model.")
    parser.add_argument("--jaw", choices=("upper", "lower"), required=True,
                         help="Which occlusal view to train a dedicated model for (frontal/lateral "
                              "views from the original paper aren't in this dataset).")
    args = parser.parse_args()

    data_yaml = write_data_yaml(args.jaw)

    model = YOLO("yolo11n-seg.pt")
    model.train(
        data=str(data_yaml),
        imgsz=1024,
        batch=16,
        epochs=2000,
        patience=300,
        project=str(NGUYEN_DIR / "runs"),
        name=args.jaw,
        # Windows' "spawn" multiprocessing start method pickles the whole worker process state
        # (comparison/ghorbani/train_detect.py hit a MemoryError from exactly this with the
        # default workers=8). workers=0 runs data loading in the main process instead.
        workers=0,
    )


if __name__ == "__main__":
    main()
