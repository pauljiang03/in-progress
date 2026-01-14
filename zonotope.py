import torch


class Zonotope:


    def __init__(self, center, generators=None):
        if center.ndim != 1:
            raise ValueError("center must be 1D")
        self.center = center
        self.d = center.numel()

        if generators is None:
            self.generators = torch.zeros((0, self.d), device=center.device, dtype=center.dtype)
        else:
            self.generators = generators

        self.m = self.generators.shape[0]

    def from_linf_ball(x, eps):
        d = x.numel()
        G = eps * torch.eye(d, device=x.device, dtype=x.dtype)
        return Zonotope(x.clone(), G)

    def zero_like(z):
        return Zonotope(torch.zeros_like(z.center))

    def clone(self):
        return Zonotope(self.center.clone(), self.generators.clone())

    def concretize(self):
        if self.m == 0:
            return self.center.clone(), self.center.clone()
        rad = torch.sum(torch.abs(self.generators), dim=0)
        return self.center - rad, self.center + rad

    def add(self, other):
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

    def affine(self, W, b=None):
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
        new_G = torch.zeros_like(G)
        extra = []

        for j in range(d):
            lj = float(l[j])
            uj = float(u[j])

            if uj <= 0:
                new_c[j] = 0
                new_G[:, j] = 0
                continue

            if lj >= 0:
                new_c[j] = c[j]
                new_G[:, j] = G[:, j]
                continue

            alpha = uj / (uj - lj)
            beta = (-lj) * alpha / 2.0
            gamma = beta

            new_c[j] = alpha * c[j] + beta
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
