"""
Stage 1 of the Ghorbani et al. reproduction: a class-agnostic YOLOv8n tooth detector, trained on
the bbox labels prepare_detect_labels.py derives from CHAI's own GT polygons (junctioned images,
no copy - see that script's docstring). Uses ultralytics' own high-level trainer directly (unlike
train_classify.py's manual loop) since the detection dataset is already in the folder format
ultralytics expects, so there's no reason to bypass its trainer here.

Hyperparameters match the paper's stated Stage 1 setup: 640x640 input, lr 1e-4, batch 64, 10
epochs.

Usage:
    .venv/Scripts/python.exe comparison/ghorbani/train_detect.py
"""
from pathlib import Path
from ultralytics import YOLO

GHORBANI_DIR = Path(__file__).resolve().parent
DATA_YAML = GHORBANI_DIR / "detect_dataset" / "data.yaml"


def main():
    if not DATA_YAML.exists():
        raise SystemExit(f"{DATA_YAML} not found - run prepare_detect_labels.py first.")

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_YAML),
        imgsz=640,
        epochs=10,
        batch=64,
        lr0=0.0001,
        project=str(GHORBANI_DIR / "runs"),
        name="detect",
        # Windows' multiprocessing "spawn" start method pickles the whole worker process state
        # (unlike Linux's copy-on-write "fork") every time a DataLoader worker starts - with the
        # default workers=8 that serialization step itself was enough to MemoryError before any
        # actual training happened. workers=0 runs data loading in the main process instead, no
        # spawning at all. Not part of the paper's spec - a runtime necessity on this machine, not
        # a hyperparameter deviation, so no need to flag it against Ghorbani's own numbers.
        workers=0,
    )


if __name__ == "__main__":
    main()
