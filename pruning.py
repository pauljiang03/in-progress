import copy
import torch


def mask_from_outgoing_weight_norm(next_linear, keep_ratio, p=1):
  
    W = next_linear.weight.data  # (out, in)
    if p == 1:
        scores = W.abs().sum(dim=0)
    elif p == 2:
        scores = torch.sqrt((W * W).sum(dim=0))
    else:
        raise ValueError("p must be 1 or 2")

    n = scores.numel()
    k = int(round(n * keep_ratio))
    k = max(1, min(k, n))  # keep at least 1, at most n

    keep_idx = torch.topk(scores, k=k, largest=True).indices
    mask = torch.zeros(n, device=W.device, dtype=torch.float32)
    mask[keep_idx] = 1.0
    return mask


def apply_mask_pruning_same_shape_mlp(model, mask_fc1=None, mask_fc2=None):
  
    m = copy.deepcopy(model)

    if mask_fc1 is not None:
        pruned = (mask_fc1 == 0)
        m.fc2.weight.data[:, pruned] = 0.0

    if mask_fc2 is not None:
        pruned = (mask_fc2 == 0)
        m.fc3.weight.data[:, pruned] = 0.0

    return m


def count_pruned(mask):
    if mask is None:
        return 0, 0
    total = int(mask.numel())
    kept = int(mask.sum().item())
    pruned = total - kept
    return pruned, total
