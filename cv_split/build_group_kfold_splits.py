"""
Feasibility check for GroupKFold CV (see chai_paper_todo.md's CV-methodology plan): hashes every
image across train+val+test (71,521 images total) with the same 8x8 average-hash used by the
earlier (now-backed-up) find_duplicate_images.py / check_train_test_leakage.py, assigns every
image a group id by exact-hash cluster, and reports the cluster-size distribution. This only
answers "can GroupKFold(k=10) even produce reasonably balanced folds given these clusters" - it
does not yet carve out a new test set or run GroupKFold itself; that's a follow-up once this
diagnostic looks sane.

Usage:
    .venv/Scripts/python.exe cv_split/build_group_kfold_splits.py
"""
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

_bootstrap_root = Path(__file__).resolve().parent.parent
if str(_bootstrap_root) not in sys.path:
    sys.path.append(str(_bootstrap_root))
from config import PathConfig
ROOT = PathConfig.ROOT
DATASET_DIR = ROOT.parent / "dataset"

HASH_SIZE = 8
SPLITS = ("train", "val", "test")
JAWS = ("upper", "lower")


def ahash(path):
    im = Image.open(path).convert("L").resize((HASH_SIZE, HASH_SIZE), Image.LANCZOS)
    arr = np.array(im, dtype=np.float32)
    avg = arr.mean()
    bits = (arr > avg).flatten()
    return bits.tobytes()


def main():
    # (split, jaw, filename, hash)
    entries = []
    for split in SPLITS:
        for jaw in JAWS:
            d = DATASET_DIR / split / "images" / jaw
            paths = sorted(d.glob("*.png"))
            print(f"Hashing {split}/{jaw}: {len(paths)} images...")
            for p in paths:
                entries.append((split, jaw, p.name, ahash(p)))
    print(f"\nTotal images hashed: {len(entries)}")

    clusters = defaultdict(list)  # hash -> [(split, jaw, filename), ...]
    for split, jaw, name, h in entries:
        clusters[h].append((split, jaw, name))

    sizes = sorted((len(v) for v in clusters.values()), reverse=True)
    n_singleton = sum(1 for s in sizes if s == 1)
    n_multi = len(sizes) - n_singleton
    n_images_in_multi = sum(s for s in sizes if s > 1)

    print(f"\n=== Cluster summary (exact-hash groups across train+val+test) ===")
    print(f"Distinct groups: {len(clusters)}")
    print(f"Singleton groups (unique image, no duplicate anywhere): {n_singleton}")
    print(f"Multi-member groups: {n_multi}")
    print(f"Images belonging to a multi-member group: {n_images_in_multi} "
          f"({100*n_images_in_multi/len(entries):.1f}% of all images)")
    print(f"Largest 20 group sizes: {sizes[:20]}")

    # How many groups cross a split boundary today (informational only - this is the leakage
    # that motivated GroupKFold in the first place, now measured over the FULL dataset rather
    # than just test-vs-{train,val} pairwise).
    cross_split_groups = 0
    for members in clusters.values():
        if len({m[0] for m in members}) > 1:
            cross_split_groups += 1
    print(f"\nGroups whose members span more than one current split: {cross_split_groups}")

    # Per-jaw group-size distribution, since GroupKFold will in practice be run separately (or
    # at least reported) per jaw - upper/lower are different model inputs downstream.
    for jaw in JAWS:
        jaw_sizes = sorted((len({m for m in v if m[1] == jaw})
                             for v in clusters.values() if any(m[1] == jaw for m in v)), reverse=True)
        jaw_sizes = [s for s in jaw_sizes if s > 0]
        n_jaw_images = sum(1 for e in entries if e[1] == jaw)
        print(f"\n--- {jaw} ---")
        print(f"Images: {n_jaw_images}, distinct groups touching this jaw: {len(jaw_sizes)}")
        print(f"Largest 10 group sizes (this jaw only): {jaw_sizes[:10]}")

    out = {
        "clusters": dict(clusters),  # hash -> [(split, jaw, filename), ...]
    }
    out_path = ROOT / "cv_split" / "group_kfold_clusters.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nSaved full cluster data to {out_path}")


if __name__ == "__main__":
    main()
