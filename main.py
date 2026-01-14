import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import MLP
from train import train_mnist
from diff_verifier import DiffVerifierMLP

from pruning import (
    mask_from_outgoing_weight_norm,
    apply_mask_pruning_same_shape_mlp,
    count_pruned,
)


def load_test_samples(device, n):
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=1, shuffle=True)
    out = []
    for i, (x, y) in enumerate(loader):
        out.append((x.to(device), int(y.item())))
        if i + 1 >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "verify_prune_mag"], default="verify_prune_mag")
    ap.add_argument("--weights", type=str, default="m1.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eps", type=float, default=0.001)
    ap.add_argument("--num_samples", type=int, default=25)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--keep_ratio_fc1", type=float, default=0.9)
    ap.add_argument("--keep_ratio_fc2", type=float, default=0.9)
    ap.add_argument("--p_norm", type=int, choices=[1, 2], default=1)

    args = ap.parse_args()
    device = torch.device(args.device)

    if args.mode == "train":
        m = MLP()
        train_mnist(m, device=device, epochs=args.epochs, save_path=args.weights)
        return

    # load base (unpruned)
    base = MLP()
    base.load_state_dict(torch.load(args.weights, map_location=device))
    base.to(device).eval()

    # prune fc1 outputs using outgoing norms in fc2
    mask_fc1 = mask_from_outgoing_weight_norm(base.fc2, keep_ratio=args.keep_ratio_fc1, p=args.p_norm)
    # prune fc2 outputs using outgoing norms in fc3
    mask_fc2 = mask_from_outgoing_weight_norm(base.fc3, keep_ratio=args.keep_ratio_fc2, p=args.p_norm)

    pruned = apply_mask_pruning_same_shape_mlp(base, mask_fc1=mask_fc1, mask_fc2=mask_fc2)
    pruned.to(device).eval()

    pr1, tot1 = count_pruned(mask_fc1)
    pr2, tot2 = count_pruned(mask_fc2)
    print(f"Pruning (VeriPrune-style magnitude):")
    print(f"  fc1 neurons pruned: {pr1}/{tot1} (keep_ratio={args.keep_ratio_fc1})")
    print(f"  fc2 neurons pruned: {pr2}/{tot2} (keep_ratio={args.keep_ratio_fc2})")
    print()

    # model1 = pruned, model2 = base
    verifier = DiffVerifierMLP(pruned, base, device=device)
    samples = load_test_samples(device, args.num_samples)

    agree = 0
    certified = 0
    for i, (x, y) in enumerate(samples):
        res = verifier.verify_one(x, eps=args.eps)
        same = (res["pred1"] == res["pred2"])
        agree += int(same)
        certified += int(res["certified_same_pred"])
        print(f"[{i:02d}] pred_pruned={res['pred1']} pred_base={res['pred2']} agree={same} certified={res['certified_same_pred']}")

    print(f"\nAgree on sampled: {agree}/{len(samples)}")
    print(f"Certified same prediction: {certified}/{len(samples)} at eps={args.eps}")


if __name__ == "__main__":
    main()
