import torch
from torch.utils.data import Dataset


class ArchSequenceDataset(Dataset):
    """Loads a cache produced by build_dataset.py: one entry per dental arch, holding a
    variable-length (<=12) sequence of tooth tokens."""

    def __init__(self, cache_path):
        self.sequences = torch.load(cache_path)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        return item["geom"], item["img_vec"], item["prob_vec"], item["target"]


def pad_collate(batch):
    """Pads a batch of variable-length arches to the batch's max tooth count.

    Returns (geom, img_vec, prob_vec, target, key_padding_mask), where target is padded
    with -100 (ignored by CrossEntropyLoss) and key_padding_mask is True at pad positions
    (the convention nn.TransformerEncoder's src_key_padding_mask expects).
    """
    lengths = [geom.size(0) for geom, _, _, _ in batch]
    max_len = max(lengths)
    geom_dim = batch[0][0].size(1)
    img_dim = batch[0][1].size(1)
    prob_dim = batch[0][2].size(1)
    batch_size = len(batch)

    geom_out = torch.zeros(batch_size, max_len, geom_dim)
    img_out = torch.zeros(batch_size, max_len, img_dim)
    prob_out = torch.zeros(batch_size, max_len, prob_dim)
    target_out = torch.full((batch_size, max_len), -100, dtype=torch.long)
    key_padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)

    for i, (geom, img_vec, prob_vec, target) in enumerate(batch):
        k = geom.size(0)
        geom_out[i, :k] = geom
        img_out[i, :k] = img_vec
        prob_out[i, :k] = prob_vec
        target_out[i, :k] = target
        key_padding_mask[i, :k] = False

    return geom_out, img_out, prob_out, target_out, key_padding_mask
