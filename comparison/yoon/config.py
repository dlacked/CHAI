"""
Config for reproducing Yoon et al.'s tooth-number-recognition model on CHAI's own dataset.

Inherits mmdetection's own stock cascade-rcnn_r101_fpn_1x_coco.py (fetched via
`mim download mmdet --config cascade-rcnn_r101_fpn_1x_coco`) - notably, its RPN anchor generator
(scales=[8], ratios=[0.5,1.0,2.0]) and its 3-stage bbox_coder target_stds ([0.1,0.1,0.2,0.2] /
[0.05,0.05,0.1,0.1] / [0.033,0.033,0.067,0.067]) already match the paper's own stated values
exactly - strong evidence the paper trained this stock config with only a handful of overrides,
which is what this file does too, rather than hand-rebuilding the architecture from scratch.

Deviations from the paper, each forced by this machine rather than a methodology choice:
  - batch_size=2 (train_dataloader below): paper used 8 samples/GPU x 2 Tesla V100 (32GB each,
    64GB total) = 16 total. This machine has one 12GB GPU; 8 (matching the paper's per-GPU count)
    OOM'd, and 4 measured worse ETA than 2 despite fewer iterations/epoch (see train_dataloader's
    own comment for the measurements), so 2 is what's actually used. optim_wrapper's lr is
    linearly scaled down to match (0.0002 * 2/16 = 0.000025 - see train.py's docstring, which
    used to incorrectly claim no scaling was applied).
  - num_workers=0 and mp_start_method='spawn': Windows has no fork() - see
    comparison/ghorbani/train_detect.py's docstring for the identical Windows/spawn reasoning.
  - LR milestones: the paper states "step-based policy with warm-up and decay phases" for 40
    epochs but never gives the exact decay epochs. Scaled proportionally from the stock 1x
    schedule's 8/12 and 11/12 fractions (milestones=[8,11] over 12 epochs) onto 40 epochs.

Classes: the 24 full two-digit FDI numbers this dataset actually has GT for - see
prepare_coco_labels.py's docstring for why (no primary teeth, no 7/8 wisdom teeth in CHAI's own
dataset, unlike Yoon et al.'s own dataset which explicitly excluded primary teeth but did include
7/8).
"""
_base_ = ['./cascade-rcnn_r101_fpn_1x_coco.py']

FDI_CLASSES = tuple(str(q * 10 + d) for q in (1, 2, 3, 4) for d in range(1, 7))
NUM_CLASSES = len(FDI_CLASSES)

DATA_ROOT = 'coco_dataset/'

# --- Model: only num_classes changes across all three cascade stages - the RPN anchors, bbox
# coders, and everything else already matches the paper's stated setup (see module docstring).
# mmengine's config inheritance replaces list-valued keys wholesale rather than merging element-
# by-element, so overriding bbox_head requires repeating each stage's full dict (copied verbatim
# from cascade-rcnn_r101_fpn_1x_coco.py, only num_classes changed) - a bare
# [dict(num_classes=24), ...] silently drops every other required key (type, bbox_coder, losses)
# and fails at model-build time with a missing "type" KeyError.
_STAGE_STDS = ([0.1, 0.1, 0.2, 0.2], [0.05, 0.05, 0.1, 0.1], [0.033, 0.033, 0.067, 0.067])
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=NUM_CLASSES,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=stds,
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
            )
            for stds in _STAGE_STDS
        ]
    )
)

# COCO-pretrained backbone+neck+RPN weights (the mim-downloaded checkpoint) - the 3 bbox_head
# stages' classification/regression layers won't shape-match (80 COCO classes vs our 24) and are
# reinitialized automatically by mmengine's checkpoint loader, matching the paper's own
# "initialized from a pretrained model trained on the COCO dataset" setup.
load_from = 'cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth'

# --- Dataset: CHAI's own COCO-format conversion (prepare_coco_labels.py), both jaws pooled per
# split, images reused via NTFS junction rather than copied.
metainfo = dict(classes=FDI_CLASSES)

train_dataloader = dict(
    batch_size=2,  # dropped back from 4 - measured on this machine, time/iter roughly doubled
    # (0.375s -> 0.79s) when batch went 2->4, so the fewer-iterations-per-epoch win was a wash
    # (ETA actually got slightly worse, 4d21h -> 5d8h) while memory usage nearly doubled (5.5GB ->
    # 10GB, uncomfortably close to this GPU's 12GB). This model is compute-bound enough per image
    # that batching harder just doesn't pay off here - 2 is both faster and safer. Effective batch
    # is now 1/8 of the paper's 16 (2 GPUs x 8) - see optim_wrapper below for the matching lr.
    num_workers=0,  # Windows has no fork() - see module docstring
    persistent_workers=False,
    dataset=dict(
        data_root=DATA_ROOT,
        metainfo=metainfo,
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/images/'),
    ),
)
val_dataloader = dict(
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        data_root=DATA_ROOT,
        metainfo=metainfo,
        ann_file='val/annotations.json',
        data_prefix=dict(img='val/images/'),
    ),
)
test_dataloader = dict(
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        data_root=DATA_ROOT,
        metainfo=metainfo,
        ann_file='test/annotations.json',
        data_prefix=dict(img='test/images/'),
    ),
)
val_evaluator = dict(ann_file=DATA_ROOT + 'val/annotations.json')
test_evaluator = dict(ann_file=DATA_ROOT + 'test/annotations.json')

# --- Optimizer: paper's stated SGD lr=0.0002 at effective batch 16 (2 GPUs x 8/GPU), linearly
# scaled down to batch_size=2 above -> 0.0002 * (2/16) = 0.000025. weight_decay=0.0000001 is the
# paper's own stated value, unaffected by batch size. Momentum not stated - kept at the stock
# config's 0.9.
optim_wrapper = dict(
    optimizer=dict(lr=0.000025, momentum=0.9, weight_decay=0.0000001)
)

# --- Schedule: 40 epochs per the paper; decay milestones scaled from the stock 1x schedule's
# 8/12, 11/12 fractions (see module docstring - the paper doesn't give exact decay epochs).
max_epochs = 40
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True,
         milestones=[27, 37], gamma=0.1),
]

# Windows has no fork() - mmengine's default mp_start_method is 'fork', which errors immediately
# on Windows. num_workers=0 above means no worker processes actually spawn, but this is set
# regardless so the runner doesn't fail trying to configure a start method Windows doesn't have.
env_cfg = dict(mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0))

# A full R101 Cascade R-CNN checkpoint is 300+ MB - interval=1 over 40 epochs would be 12+ GB.
# Keep only the 3 most recent plus whichever epoch had the best val bbox mAP.
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, save_best='coco/bbox_mAP_50')
)

work_dir = 'runs/'

# Resume handling lives in train.py, not here: mmengine's Runner.load_or_resume() treats
# `load_from` itself as the resume target whenever `resume=True` is also set (see
# `elif self._resume and self._load_from is not None: resume_from = self._load_from` in
# mmengine/runner/runner.py) - since this config's load_from is the COCO-pretrained *weights-only*
# release checkpoint (no optimizer/message_hub/param_scheduler state), setting resume=True here
# unconditionally makes every run - including the very first one - try to "resume" from that
# weights-only file and crash with `KeyError: 'message_hub'`. train.py instead checks whether a
# real in-progress checkpoint already exists in work_dir and only enables resume+clears load_from
# when one does.
