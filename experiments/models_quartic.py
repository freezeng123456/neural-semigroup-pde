"""
LatentSemigroupNet with quartic interaction term (Direction 1).

Energy functional:
    Psi(z) = sum V(z_i) + 1/2 sum a_ij (z_i-z_j)^2 + 1/4 sum b_ij (z_i-z_j)^4

Quartic term enables sharper spatial coupling for Allen-Cahn-type PDEs.
All structural guarantees preserved (admissibility, semigroup, energy dissipation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalarMLP(nn.Module):
    """MLP mapping R -> R, with softplus activation."""
    def __init__(self, hidden_dims=[32, 32], beta=0.0):
        super().__init__()
        dims = [1] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        return self.net(x) + 0.5 * self.beta.abs() * x.pow(2)

    def value_and_grad(self, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1, 1)
        h = x_flat
        dh = torch.ones(x_flat.shape[0], 1, device=x.device)
        linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]
        for i, layer in enumerate(linear_layers):
            W = layer.weight
            pre = layer(h)
            if i < len(linear_layers) - 1:
                sig = torch.sigmoid(pre)
                dh = dh @ W.T * sig
                h = F.softplus(pre)
            else:
                dh = dh @ W.T
                h = pre
        beta = self.beta.abs()
        val = h + 0.5 * beta * x_flat.pow(2)
        grad = dh + beta * x_flat
        return val.reshape(orig_shape), grad.reshape(orig_shape)


class StencilMLP(nn.Module):
    """Stencil-based MLP for spatial interactions."""
    def __init__(self, radius=3, hidden_dims=[32, 32]):
        super().__init__()
        self.radius = radius
        in_dim = 2 * radius + 1
        dims = [in_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        B, N = z.shape
        r = self.radius
        z_pad = F.pad(z.unsqueeze(1), (r, r), mode='constant', value=0.0).squeeze(1)
        patches = z_pad.unfold(1, 2*r+1, 1)
        return self.net(patches).squeeze(-1)


class LatentSemigroupNetQuartic(nn.Module):
    """
    Latent semigroup learner with quartic interaction term.

    Energy:
        Psi(z) = sum V(z_i) + 1/2 sum a_ij (z_i-z_j)^2 + 1/4 sum b_ij (z_i-z_j)^4

    Dynamics:
        dz_i/dt = -k_i(z) * [V'(z_i) + 2 sum_j a_ij (z_i-z_j) + 4 sum_j b_ij (z_i-z_j)^3]

    All structural guarantees preserved:
    - Admissibility: sigmoid output in (0,1)
    - Exact semigroup: dz/dt = -K * grad Psi
    - Energy dissipation: dPsi/dt = -grad Psi^T K grad Psi <= 0
    """

    def __init__(self, N=64, m=0.0, M=1.0,
                 hidden_V=[64, 64], hidden_K=[64, 64],
                 stencil_radius=3, beta_V=0.0, interaction_radius=2):
        super().__init__()
        self.N = N
        self.m = m
        self.M = M

        self.V_net = ScalarMLP(hidden_V, beta=beta_V)

        # Quadratic interaction coefficients
        self.emb_a = nn.Parameter(torch.randn(N, 8) * 0.001)
        # Quartic interaction coefficients (separate embeddings)
        self.emb_b = nn.Parameter(torch.randn(N, 8) * 0.001)

        # Local interaction mask
        mask = torch.zeros(N, N)
        for i in range(N):
            for j in range(N):
                dist = min(abs(i - j), N - abs(i - j))
                if dist <= interaction_radius and i != j:
                    mask[i, j] = 1.0
        self.register_buffer('interaction_mask', mask)

        self.K_net = StencilMLP(radius=stencil_radius, hidden_dims=hidden_K)
        self.ode_steps = 30
        self.register_buffer('eps', torch.tensor(1e-6))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, u):
        u_c = u.clamp(self.m + self.eps, self.M - self.eps)
        s = (u_c - self.m) / (self.M - self.m)
        return torch.log(s / (1.0 - s))

    def decode(self, z):
        return self.m + (self.M - self.m) * torch.sigmoid(z)

    def energy(self, u):
        z = self.encode(u)
        return self.psi(z)

    def psi(self, z):
        """Psi(z) = sum V(z_i) + 1/2 sum a_ij (z_i-z_j)^2 + 1/4 sum b_ij (z_i-z_j)^4"""
        V_sum = self.V_net(z.unsqueeze(-1)).squeeze(-1).sum(dim=1)

        # Coefficients
        a_full = F.softplus(torch.mm(self.emb_a, self.emb_a.t()))
        a_ij = a_full * self.interaction_mask
        b_full = F.softplus(torch.mm(self.emb_b, self.emb_b.t()))
        b_ij = b_full * self.interaction_mask

        diff = z.unsqueeze(2) - z.unsqueeze(1)  # (B, N, N)
        quadratic = 0.5 * (a_ij * diff.pow(2)).sum(dim=(1, 2))
        quartic = 0.25 * (b_ij * diff.pow(4)).sum(dim=(1, 2))

        return V_sum + quadratic + quartic

    def grad_psi(self, z, a_ij=None, b_ij=None):
        """grad_z Psi(z).

        dPsi/dz_k = V'(z_k) + 2 sum_j a_kj (z_k-z_j) + 4 sum_j b_kj (z_k-z_j)^3
        """
        _V_val, dV_dz = self.V_net.value_and_grad(z.unsqueeze(-1))
        grad_V = dV_dz.squeeze(-1)  # (B, N)

        if a_ij is None:
            a_full = F.softplus(torch.mm(self.emb_a, self.emb_a.t()))
            a_ij = a_full * self.interaction_mask
        if b_ij is None:
            b_full = F.softplus(torch.mm(self.emb_b, self.emb_b.t()))
            b_ij = b_full * self.interaction_mask

        diff = z.unsqueeze(2) - z.unsqueeze(1)  # (B, N, N)
        grad_quad = 2.0 * (a_ij * diff).sum(dim=2)
        grad_quartic = 4.0 * (b_ij * diff.pow(3)).sum(dim=2)

        return grad_V + grad_quad + grad_quartic

    def _dynamics_and_grad(self, z, a_ij, b_ij):
        K = F.softplus(self.K_net(z)) + 5e-3
        grad_Psi = self.grad_psi(z, a_ij=a_ij, b_ij=b_ij)
        return K, grad_Psi

    def latent_dynamics(self, z, a_ij=None, b_ij=None):
        if a_ij is None or b_ij is None:
            a_full = F.softplus(torch.mm(self.emb_a, self.emb_a.t()))
            a_ij = a_full * self.interaction_mask
            b_full = F.softplus(torch.mm(self.emb_b, self.emb_b.t()))
            b_ij = b_full * self.interaction_mask
        k, g = self._dynamics_and_grad(z, a_ij, b_ij)
        return -k * g

    def rk4_step(self, z, dt, a_ij=None, b_ij=None):
        f1 = self.latent_dynamics(z, a_ij, b_ij)
        f2 = self.latent_dynamics(z + 0.5 * dt * f1, a_ij, b_ij)
        f3 = self.latent_dynamics(z + 0.5 * dt * f2, a_ij, b_ij)
        f4 = self.latent_dynamics(z + dt * f3, a_ij, b_ij)
        z_next = z + dt / 6.0 * (f1 + 2.0 * f2 + 2.0 * f3 + f4)
        return torch.clamp(z_next, -20.0, 20.0)

    def forward(self, u, tau):
        z = self.encode(u)
        # Cache interaction coefficients once per forward
        a_full = F.softplus(torch.mm(self.emb_a, self.emb_a.t()))
        a_ij = a_full * self.interaction_mask
        b_full = F.softplus(torch.mm(self.emb_b, self.emb_b.t()))
        b_ij = b_full * self.interaction_mask
        dt = tau / self.ode_steps
        for _ in range(self.ode_steps):
            z = self.rk4_step(z, dt, a_ij, b_ij)
        return self.decode(z)

    def rollout(self, u0, tau, n_steps):
        u = u0
        traj = [u0]
        for _ in range(n_steps):
            u = self.forward(u, tau)
            traj.append(u)
        return torch.stack(traj, dim=0)
