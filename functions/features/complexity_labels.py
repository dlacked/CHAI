"""
Loads each test-set arch's dataset-provided Arch Complexity label (Class I/II/III malocclusion
severity, from metadata.json) - the ground-truth label used throughout the paper's Malocclusion
(Class II+III) subset evaluation (Table III).

Moved out of the now-removed Transformer/complexity/ directory (which otherwise only held an
unused Arch Complexity Transformer model/training pipeline) - this is the one piece of that
directory any live script actually depends on.
"""
import json
from pathlib import Path


def load_complexity_labels(dataset_dir, split):
    meta_path = Path(dataset_dir) / split / "metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        info = json.load(f)["info"]
    return {Path(e["image_filepath"]).name: e["complexity"] for e in info}
