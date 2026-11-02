"""
Defines ToothPositionClassifier: the base ResNet18 + 5-dim meta-vector (x1,y1,x2,y2,theta)
architecture every pooled/comp variant in this project inherits from or mirrors (see
train_mirrored.py's ToothPositionClassifierNoTheta, train_chai.py, train_chai_backbone.py).
Still imported dynamically by server.py for its own (currently unused - no lower_best.pth/
upper_best.pth exist) per-jaw inference path.

This file's own standalone training pipeline (a per-jaw, 6-way digit classifier with the full
[x1,y1,x2,y2,theta] meta vector - the original single-jaw baseline before the project moved to
pooled both-jaws models) was removed 2026-10-30 as dead code: nothing re-trains this baseline
anymore (its own model files don't exist on disk), and nothing else in the codebase called
ToothDataset or train_model. See git history before this date if this baseline ever needs to be
retrained from scratch.
"""
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision import models

_bootstrap_root = Path(__file__).resolve().parent.parent
if str(_bootstrap_root) not in sys.path:
    sys.path.append(str(_bootstrap_root))
from config import PathConfig
project_root = PathConfig.ROOT
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


class ToothPositionClassifier(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(ToothPositionClassifier, self).__init__()
        # ImageNet-pretrained backbone for training from scratch. Inference-only callers (e.g.
        # server.py, which immediately overwrites every weight via load_state_dict) should pass
        # pretrained=False to skip this fetch entirely - it's wasted network/disk I/O when the
        # weights are about to be discarded anyway.
        if not pretrained:
            self.resnet = models.resnet18(weights=None)
        else:
            try:
                self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                print("Pretrained ResNet18 weights loaded.")
            except (ImportError, AttributeError):
                self.resnet = models.resnet18(pretrained=True)
                print("Pretrained ResNet18 weights loaded (legacy).")

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove classifier head

        # Meta-feature MLP
        self.meta_fc = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU()
        )

        # Combined classifier head (now 6 output classes by default)
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, meta):
        img_features = self.resnet(img)
        meta_features = self.meta_fc(meta)
        combined = torch.cat((img_features, meta_features), dim=1)
        logits = self.classifier(combined)
        return torch.softmax(logits, dim=1)
