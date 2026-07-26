"""Split dataset/metadata.json (complexity per image) into per-split metadata.json
files, matched against the images already present under train/val/test/images/<jaw>/.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT.parent / "dataset"
SPLITS = ("train", "val", "test")
JAWS = ("lower", "upper")


def main():
    with open(DATASET_DIR / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)["info"]

    basename_to_complexity = {Path(e["image_filepath"]).name: e["complexity"] for e in meta}

    for split in SPLITS:
        entries = []
        unmatched = 0
        for jaw in JAWS:
            jaw_dir = DATASET_DIR / split / "images" / jaw
            for img_path in sorted(jaw_dir.glob("*.png")):
                complexity = basename_to_complexity.get(img_path.name)
                if complexity is None:
                    unmatched += 1
                    continue
                entries.append({
                    "image_filepath": f"images/{jaw}/{img_path.name}",
                    "complexity": complexity,
                })

        out_path = DATASET_DIR / split / "metadata.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"info": entries}, f, ensure_ascii=False, indent=2)

        print(f"{split}: {len(entries)} entries written to {out_path} ({unmatched} unmatched)")


if __name__ == "__main__":
    main()
