import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import MLP
from train import train_mnist
from diff_verifier import DiffVerifierMLP
from pruning import make_random_mask, apply_mask_pruning_same_shape_mlp


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
    ap.add_argument("--mode", choices=["train", "verify_diff", "verify_prune_mask"], default="verify_prune_mask")
    ap.add_argument("--weights", type=str, default="m1.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eps", type=float, default=0.003)
    ap.add_argument("--num_samples", type=int, default=25)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--keep_ratio_fc1", type=float, default=0.5)
    ap.add_argument("--keep_ratio_fc2", type=float, default=0.5)
    ap.add_argument("--prune_seed", type=int, default=0)

    args = ap.parse_args()
    device = torch.device(args.device)

    if args.mode == "train":
        m = MLP()
        train_mnist(m, device=device, epochs=args.epochs, save_path=args.weights)
        return

    base = MLP()
    base.load_state_dict(torch.load(args.weights, map_location=device))
    base.to(device).eval()

    if args.mode == "verify_diff":
        raise SystemExit("Use --mode verify_prune_mask for mask-pruned vs unpruned.")

    torch.manual_seed(args.prune_seed)
    mask1 = make_random_mask(base.fc1.out_features, args.keep_ratio_fc1, device=device)
    mask2 = make_random_mask(base.fc2.out_features, args.keep_ratio_fc2, device=device)

    pruned = apply_mask_pruning_same_shape_mlp(base, mask_fc1=mask1, mask_fc2=mask2)
    pruned.to(device).eval()

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
    print(f"Masks: keep_ratio_fc1={args.keep_ratio_fc1}, keep_ratio_fc2={args.keep_ratio_fc2}, prune_seed={args.prune_seed}")


if __name__ == "__main__":
    main()
