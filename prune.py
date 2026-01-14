# pruning.py
import copy
import torch


def make_random_mask(n, keep_ratio, device=None):
    k = int(round(n * keep_ratio))
    idx = torch.randperm(n, device=device)
    mask = torch.zeros(n, device=device, dtype=torch.float32)
    mask[idx[:k]] = 1.0
    return mask


def apply_mask_pruning_same_shape_mlp(model, mask_fc1=None, mask_fc2=None):
    m = copy.deepcopy(model)

    if mask_fc1 is not None:
        if mask_fc1.ndim != 1 or mask_fc1.numel() != m.fc1.out_features:
            raise ValueError("mask_fc1 must be shape (hidden,) matching fc1 out_features")
        pruned = (mask_fc1 == 0)
        m.fc2.weight.data[:, pruned] = 0.0

    if mask_fc2 is not None:
        if mask_fc2.ndim != 1 or mask_fc2.numel() != m.fc2.out_features:
            raise ValueError("mask_fc2 must be shape (hidden,) matching fc2 out_features")
        pruned = (mask_fc2 == 0)
        m.fc3.weight.data[:, pruned] = 0.0

    return m
