import torch
import torch.nn as nn
from zonotope import Zonotope


class DiffVerifierMLP:
    """
      - Zpp tracks reachable set through model2
      - ZΔ tracks (model1 - model2) difference reachable set
      - Both start from the same input zonotope; Δ starts at 0

      If z1 = z2 + Δz, then
        W1 z1 + b1 - (W2 z2 + b2) = W1 Δz + (W1-W2) z2 + (b1-b2)
      So:
        ZΔ_out = Affine(W1, ZΔ_in) + Affine(W1-W2, Zpp_in) + (b1-b2)

      Zpp := ReLU(Zpp)
      Zp  := ReLU(Zpp + ZΔ) 
      ZΔ := Zp - Zpp
    """

    def __init__(self, model1: nn.Module, model2: nn.Module, device: torch.device):
        self.m1 = model1.to(device).eval()
        self.m2 = model2.to(device).eval()
        self.device = device

        self.l1 = [self.m1.fc1, self.m1.fc2, self.m1.fc3]
        self.l2 = [self.m2.fc1, self.m2.fc2, self.m2.fc3]
        for a, b in zip(self.l1, self.l2):
            if not (isinstance(a, nn.Linear) and isinstance(b, nn.Linear)):
                raise ValueError("Models must have fc1/fc2/fc3 Linear layers.")

    def _affine_delta(self, Zpp: Zonotope, ZD: Zonotope, W1, b1, W2, b2):
        # ZD_out = W1*ZD + (W1-W2)*Zpp + (b1-b2)
        part1 = ZD.affine(W1, None)
        part2 = Zpp.affine(W1 - W2, None)
        out = part1.add(part2)
        out = Zonotope(out.center + (b1 - b2), out.generators)
        return out

    def propagate(self, Zin: Zonotope):
        Zpp = Zin.clone()               
        ZD = Zonotope.zero_like(Zin)    

        for (L1, L2) in zip(self.l1, self.l2):
            W1, b1 = L1.weight, L1.bias
            W2, b2 = L2.weight, L2.bias

            ZD = self._affine_delta(Zpp, ZD, W1, b1, W2, b2)
            Zpp = Zpp.affine(W2, b2)

            if L1 is not self.l1[-1]:
                Zpp_relu = Zpp.relu()
                Zp_approx = Zpp.add(ZD).relu()
                ZD = Zp_approx.sub(Zpp_relu)
                Zpp = Zpp_relu

        return Zpp, ZD 

    def certify_same_argmax(Zpp_logits: Zonotope, ZD_logits: Zonotope, label: int) -> bool:
        l2, u2 = Zpp_logits.concretize()
        ld, ud = ZD_logits.concretize()

        # margin2 lower bound: l2[label] - u2[k]
        # margind lower bound: ld[label] - ud[k]
        for k in range(l2.numel()):
            if k == label:
                continue
            margin2_lb = float((l2[label] - u2[k]).item())
            margind_lb = float((ld[label] - ud[k]).item())
            if margin2_lb + margind_lb <= 0.0:
                return False
        return True
    def verify_one(self, x: torch.Tensor, eps: float) -> dict:
        x = x.to(self.device)
        x_flat = x.view(-1)

        with torch.no_grad():
            log2 = self.m2(x)
            pred2 = int(log2.argmax(1).item())
            log1 = self.m1(x)
            pred1 = int(log1.argmax(1).item())

        Zin = Zonotope.from_linf_ball(x_flat, eps)
        Zpp_logits, ZD_logits = self.propagate(Zin)

        certified = self.certify_same_argmax(Zpp_logits, ZD_logits, label=pred2)
        return {"pred1": pred1, "pred2": pred2, "certified_same_pred": certified}
