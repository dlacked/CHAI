"""
Defines ToothPositionClassifierNoTheta, the ResNet18 backbone/classifier-head architecture
(from train.py's ToothPositionClassifier) with a 4-input meta branch (x1,y1,x2,y2, no theta)
instead of train.py's 5-input version. Used as the base architecture for CHAI (train_chai.py,
train_chai_groupkfold.py) and for scoring the Mirrored Variant checkpoint kept in
backups/superseded_20260831/ResNet_tooth_model_mirrored/ (functions/graph/tooth.py's
build_mirrored_arch_sequences dynamically loads this file for the class definition only).

This file's own standalone training pipeline (the Mirrored Variant - a combined, no-rotation,
no-theta, 6-way digit classifier; paper ablation label, retired when CHAI was redefined to the
unified 12-way, unmirrored architecture) was removed 2026-10-30 since it was dead code - nothing
re-trains this variant anymore, and its last trained checkpoint is preserved as-is in backups/.
See git history before this date for the removed ToothDatasetCombined/train_model/main code if
that variant ever needs to be retrained from scratch.
"""
import sys
from pathlib import Path

import torch.nn as nn

tooth_dir = Path(__file__).resolve().parent
if str(tooth_dir) not in sys.path:
    sys.path.append(str(tooth_dir))
from train import ToothPositionClassifier

_bootstrap_root = Path(__file__).resolve().parent.parent
if str(_bootstrap_root) not in sys.path:
    sys.path.append(str(_bootstrap_root))
from config import PathConfig
project_root = PathConfig.ROOT
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


class ToothPositionClassifierNoTheta(ToothPositionClassifier):
    """ToothPositionClassifier with a 4-dim meta input (x1,y1,x2,y2 - no theta). Theta was
    dropped from this ablation's features entirely (see main_mirrored.py's docstring: it's
    PCA-angle-derived and was shown about as fragile to missing teeth as the coordinate rotation
    itself), so the meta_fc input layer shrinks from 5 to 4 - everything else (ResNet backbone,
    classifier head) is inherited unchanged."""

    def __init__(self, num_classes=6, pretrained=True):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        self.meta_fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU()
        )
