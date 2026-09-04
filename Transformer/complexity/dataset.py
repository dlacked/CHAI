import torch
from torch.utils.data import Dataset


class ArchComplexityDataset(Dataset):
    """Loads a cache produced by build_dataset.py (no theta, [x1,y1,x2,y2]) or
    build_dataset_theta.py (with theta, [x1,y1,x2,y2,theta]): one entry per dental arch,
    holding a variable-length (<=12) sequence of per-tooth geometry tokens and a single
    arch-level complexity label. Token width is read from the data itself (see pad_collate
    below), so this class works unmodified for either cache."""

    def __init__(self, cache_path):
        self.sequences = torch.load(cache_path)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        return item["geom"], item["complexity"]


def pad_collate(batch):
    """Pads a batch of variable-length arches to the batch's max tooth count.

    Returns (geom, target, key_padding_mask). target is one scalar label per arch (not
    per-token), so it needs no padding. key_padding_mask is True at pad positions (the
    convention nn.TransformerEncoder's src_key_padding_mask expects).
    """
    lengths = [geom.size(0) for geom, _ in batch]
    max_len = max(lengths)
    geom_dim = batch[0][0].size(1)
    batch_size = len(batch)

    geom_out = torch.zeros(batch_size, max_len, geom_dim)
    target_out = torch.zeros(batch_size, dtype=torch.long)
    key_padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)

    for i, (geom, target) in enumerate(batch):
        k = geom.size(0)
        geom_out[i, :k] = geom
        target_out[i] = target
        key_padding_mask[i, :k] = False

    return geom_out, target_out, key_padding_mask
