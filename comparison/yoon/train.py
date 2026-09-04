"""
Trains the Yoon et al. reproduction (Cascade R-CNN, ResNet-101+FPN, via mmdetection/mmengine) on
CHAI's own COCO-format dataset - see config.py for the full hyperparameter mapping from the
paper's stated setup, and prepare_coco_labels.py for the data conversion this depends on.

Linear-scaling-rule LR adjustment WAS applied for the smaller batch_size (2, vs. the paper's 16
total, itself forced by this machine's single 12GB GPU vs. the paper's 2x Tesla V100 32GB) - see
config.py's optim_wrapper: lr=0.000025 (paper's 0.0002 * 2/16), not the paper's own 0.0002 as an
earlier draft of this docstring (and this comment) used to claim. This file has no lr override of
its own - it just loads config.py as-is, so config.py's value is the actual one used to train.

Usage:
    .venv/Scripts/python.exe comparison/yoon/prepare_coco_labels.py   # once, before this
    .venv/Scripts/python.exe comparison/yoon/train.py
"""
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner, find_latest_checkpoint

YOON_DIR = Path(__file__).resolve().parent


def main():
    cfg = Config.fromfile(str(YOON_DIR / 'config.py'))
    cfg.work_dir = str(YOON_DIR / 'runs')
    # Config.fromfile resolves relative paths (data_root, load_from) against the CWD the script
    # is run from, not config.py's own directory - anchor them here so this works regardless of
    # where train.py is invoked from (matches prepare_detect_labels.py's data.yaml `path:` note).
    cfg.train_dataloader.dataset.data_root = str(YOON_DIR / 'coco_dataset') + '/'
    cfg.val_dataloader.dataset.data_root = str(YOON_DIR / 'coco_dataset') + '/'
    cfg.test_dataloader.dataset.data_root = str(YOON_DIR / 'coco_dataset') + '/'
    cfg.val_evaluator.ann_file = str(YOON_DIR / 'coco_dataset' / 'val' / 'annotations.json')
    cfg.test_evaluator.ann_file = str(YOON_DIR / 'coco_dataset' / 'test' / 'annotations.json')
    cfg.load_from = str(YOON_DIR / cfg.load_from)

    # See config.py's docstring for why this branches rather than just setting cfg.resume = True
    # unconditionally: a real in-progress checkpoint (epoch_N.pth, full training state) already
    # in work_dir means resume from THAT and ignore the COCO-pretrained weights-only load_from;
    # no such checkpoint (first run, or a fresh work_dir) means the opposite - start from
    # load_from as a normal weights-only init, resume off.
    latest = find_latest_checkpoint(cfg.work_dir)
    if latest is not None:
        print(f'Found in-progress checkpoint {latest} - resuming from it.')
        cfg.resume = True
        cfg.load_from = None
    else:
        print('No in-progress checkpoint found - starting fresh from the COCO-pretrained backbone.')
        cfg.resume = False

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
