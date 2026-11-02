"""
Finalizes the GroupKFold CV split (see chai_paper_todo.md's CV-methodology plan) into a stable,
reusable per-image assignment table, reading build_group_kfold_splits.py's group_kfold_clusters.pkl
(same seed=42 greedy group-to-test assignment, same deterministic GroupKFold(k=10) over the pool)
and saving, for every one of the 71,520 images in train+val+test, which CV role it now plays:
    - role="test"            -> the new held-out test set (7,152 images, fixed across all folds)
    - role="pool", fold=k     -> one of GroupKFold's 10 folds within the remaining 64,368-image pool

Also records orig_split (the split name - train/val/test - the image file currently physically
lives under on disk) so training/eval code can still find the actual file without moving
anything on disk; only the CV role changes, not image locations.

Output: cv_split/cv_split_assignment.csv (one row per image) and cv_split/cv_split_assignment.pkl
(same data, plus the group_id, for programmatic use).

Usage:
    .venv/Scripts/python.exe cv_split/finalize_group_kfold_split.py
"""
import csv
import pickle
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

_bootstrap_root = Path(__file__).resolve().parent.parent
if str(_bootstrap_root) not in sys.path:
    sys.path.append(str(_bootstrap_root))
from config import PathConfig
ROOT = PathConfig.ROOT

TARGET_TEST_SIZE = 7152
N_FOLDS = 10
SEED = 42


def main():
    with open(ROOT / "cv_split" / "group_kfold_clusters.pkl", "rb") as f:
        data = pickle.load(f)
    clusters = data["clusters"]

    rows = []  # (split, jaw, filename, group_id)
    for gid, (h, members) in enumerate(clusters.items()):
        for split, jaw, name in members:
            rows.append((split, jaw, name, gid))
    print(f"Total images: {len(rows)}, total groups: {len(clusters)}")

    rng = random.Random(SEED)
    all_gids = list(range(len(clusters)))
    rng.shuffle(all_gids)

    group_size = {}
    for _, _, _, gid in rows:
        group_size[gid] = group_size.get(gid, 0) + 1

    test_gids = set()
    test_count = 0
    for gid in all_gids:
        if test_count >= TARGET_TEST_SIZE:
            break
        test_gids.add(gid)
        test_count += group_size[gid]
    print(f"New TEST set: {test_count} images across {len(test_gids)} groups")

    pool_rows = [r for r in rows if r[3] not in test_gids]
    test_rows = [r for r in rows if r[3] in test_gids]

    pool_groups = np.array([r[3] for r in pool_rows])
    gkf = GroupKFold(n_splits=N_FOLDS)
    X_dummy = np.zeros((len(pool_rows), 1))

    row_fold = {}  # index into pool_rows -> fold id
    for fold_id, (_, val_idx) in enumerate(gkf.split(X_dummy, groups=pool_groups)):
        for idx in val_idx:
            row_fold[idx] = fold_id

    assignment = []  # (orig_split, jaw, filename, group_id, role, fold)
    for split, jaw, name, gid in test_rows:
        assignment.append((split, jaw, name, gid, "test", -1))
    for i, (split, jaw, name, gid) in enumerate(pool_rows):
        assignment.append((split, jaw, name, gid, "pool", row_fold[i]))

    assert len(assignment) == len(rows)

    csv_path = ROOT / "cv_split" / "cv_split_assignment.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["orig_split", "jaw", "image_name", "group_id", "role", "fold"])
        w.writerows(assignment)
    print(f"Saved {csv_path} ({len(assignment)} rows)")

    pkl_path = ROOT / "cv_split" / "cv_split_assignment.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(assignment, f)
    print(f"Saved {pkl_path}")

    # Sanity reprint of fold balance.
    fold_sizes = {}
    for *_r, role, fold in assignment:
        if role == "pool":
            fold_sizes[fold] = fold_sizes.get(fold, 0) + 1
    print(f"\nFold sizes: {sorted(fold_sizes.items())}")
    print(f"Test size: {sum(1 for *_r, role, _f in assignment if role == 'test')}")


if __name__ == "__main__":
    main()
