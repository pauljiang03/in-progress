import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import MLP
from train import train_mnist
from diff_verifier import DiffVerifierMLP


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
    ap.add_argument("--mode", choices=["train", "verify_diff"], default="verify_diff")
    ap.add_argument("--weights1", type=str, default="m1.pt")
    ap.add_argument("--weights2", type=str, default="m2.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--num_samples", type=int, default=25)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    if args.mode == "train":
        m = MLP()
        train_mnist(m, device=device, epochs=args.epochs, save_path=args.weights1)
        torch.save(m.state_dict(), args.weights2)
        print(f"Also saved copy to {args.weights2}")
        return

    m1 = MLP()
    m2 = MLP()
    m1.load_state_dict(torch.load(args.weights1, map_location=device))
    m2.load_state_dict(torch.load(args.weights2, map_location=device))
    m1.to(device).eval()
    m2.to(device).eval()

    verifier = DiffVerifierMLP(m1, m2, device=device)

    samples = load_test_samples(device, args.num_samples)

    agree = 0
    certified = 0
    for i, (x, y) in enumerate(samples):
        res = verifier.verify_one(x, eps=args.eps)
        same = (res["pred1"] == res["pred2"])
        agree += int(same)
        certified += int(res["certified_same_pred"])
        print(f"[{i:02d}] pred1={res['pred1']} pred2={res['pred2']} agree={same} certified={res['certified_same_pred']}")

    print(f"\nAgree on sampled: {agree}/{len(samples)}")
    print(f"Certified same prediction: {certified}/{len(samples)} at eps={args.eps}")


if __name__ == "__main__":
    main()
