import torch


class Zonotope:

    def __init__(self, center: torch.Tensor, generators=None):
        if center.ndim != 1:
            raise ValueError(f"center must be 1D, got {tuple(center.shape)}")
        self.center = center
        d = center.numel()
        if generators is None:
            self.generators = torch.zeros((0, d), device=center.device, dtype=center.dtype)
        else:
            if generators.ndim != 2 or generators.shape[1] != d:
                raise ValueError(f"generators must be (m,{d}), got {tuple(generators.shape)}")
            self.generators = generators

    def d(self):
        return self.center.numel()

    def m(self):
        return self.generators.shape[0]

    def clone(self):
        return Zonotope(self.center.clone(), self.generators.clone())

    def from_linf_ball(x: torch.Tensor, eps: float):
        if x.ndim != 1:
            raise ValueError(f"x must be 1D, got {tuple(x.shape)}")
        d = x.numel()
        G = eps * torch.eye(d, device=x.device, dtype=x.dtype)
        return Zonotope(x.clone(), G)

    def zero_like(z):
        return Zonotope(torch.zeros_like(z.center), None)

    def concretize(self):
        if self.m == 0:
            return self.center.clone(), self.center.clone()
        rad = torch.sum(torch.abs(self.generators), dim=0)
        return self.center - rad, self.center + rad

    def add(self, other):
        if self.d != other.d:
            raise ValueError("add: dimension mismatch")
        c = self.center + other.center
        if self.m == 0 and other.m == 0:
            G = torch.zeros((0, self.d), device=c.device, dtype=c.dtype)
        elif self.m == 0:
            G = other.generators.clone()
        elif other.m == 0:
            G = self.generators.clone()
        else:
            G = torch.cat([self.generators, other.generators], dim=0)
        return Zonotope(c, G)

    def neg(self):
        return Zonotope(-self.center, -self.generators)

    def sub(self, other):
        return self.add(other.neg())

    def affine(self, W: torch.Tensor, b: torch.Tensor | None = None):
        if W.ndim != 2 or W.shape[1] != self.d:
            raise ValueError(f"W must be (out,{self.d}), got {tuple(W.shape)}")
        c = W @ self.center
        if b is not None:
            c = c + b
        if self.m == 0:
            G = torch.zeros((0, W.shape[0]), device=c.device, dtype=c.dtype)
        else:
            G = self.generators @ W.T
        return Zonotope(c, G)

    def relu(self):

        l, u = self.concretize()
        c = self.center
        G = self.generators
        d = self.d

        new_c = torch.zeros_like(c)
        new_G = torch.zeros_like(G) if self.m > 0 else torch.zeros((0, d), device=c.device, dtype=c.dtype)
        extra = []

        for j in range(d):
            lj = float(l[j].item())
            uj = float(u[j].item())

            if uj <= 0.0:
                new_c[j] = 0.0
                if self.m > 0:
                    new_G[:, j] = 0.0
                continue

            if lj >= 0.0:
                new_c[j] = c[j]
                if self.m > 0:
                    new_G[:, j] = G[:, j]
                continue

            
            alpha = uj / (uj - lj)
            beta = (-lj) * alpha / 2.0
            gamma = beta

            new_c[j] = alpha * c[j] + beta
            if self.m > 0:
                new_G[:, j] = alpha * G[:, j]

            eg = torch.zeros(d, device=c.device, dtype=c.dtype)
            eg[j] = gamma
            extra.append(eg)

        if len(extra) > 0:
            extra_G = torch.stack(extra, dim=0)
            if new_G.shape[0] == 0:
                out_G = extra_G
            else:
                out_G = torch.cat([new_G, extra_G], dim=0)
        else:
            out_G = new_G

        return Zonotope(new_c, out_G)
