"""
LatentSemigroupNet with spatially-conditioned potential V_θ (Direction 2).

Key change: V_θ(z_i) → V_θ(z_{i-r}, ..., z_i, ..., z_{i+r})
V_θ is now a StencilMLP that takes local neighborhood as input.

This allows the potential to depend on spatial context (e.g., whether z_i
is near an interface in Allen-Cahn), which the pointwise ScalarMLP cannot.

Energy functional:
    Psi(z) = sum_i V_theta(stencil_i(z)) + 1/2 sum_{i,j} a_ij (z_i-z_j)^2

Structural guarantees preserved:
- Admissibility: sigmoid output in (m,M)
- Energy dissipation: dPsi/dt = -grad Psi^T K grad Psi <= 0
- Exact semigroup: dz/dt = -K * grad Psi

grad Psi is computed via autograd (not analytical) since V now depends
on multiple z components through the stencil.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StencilVNet(nn.Module):
    """
    Stencil-based potential: V_theta(z_{i-r}, ..., z_i, ..., z_{i+r}).

    Takes local patch of radius r around each site, outputs scalar per site.
    Architecture: Linear -> Softplus -> ... -> Linear (output dim 1 per site)
    Plus optional quadratic term beta/2 * z_i^2.
    """
    def __init__(self, radius=2, hidden_dims=[64, 64], beta=0.0):
        super().__init__()
        self.radius = radius
        in_dim = 2 * radius + 1
        dims = [in_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, z):
        """
        z: (B, N) -> V: (B, N) scalar potential per site.
        Uses periodic padding.
        """
        B, N = z.shape
        r = self.radius
        # Periodic padding
        z_pad = torch.cat([z[:, -r:], z, z[:, :r]], dim=1)  # (B, N+2r)
        # Extract patches: (B, N, 2r+1)
        patches = z_pad.unfold(1, 2 * r + 1, 1)
        # MLP per site
        V = self.net(patches).squeeze(-1)  # (B, N)
        # Add quadratic term (center value)
        z_center = patches[:, :, r]  # (B, N)
        V = V + 0.5 * self.beta.abs() * z_center.pow(2)
        return V


class StencilMLP(nn.Module):
    """Stencil-based MLP for spatial interactions."""
    def __init__(self, radius=3, hidden_dims=[32, 32]):
        super().__init__()
        self.radius = radius
        in_dim = 2 * radius + 1
        dims = [in_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        B, N = z.shape
        r = self.radius
        z_pad = F.pad(z.unsqueeze(1), (r, r), mode='circular').squeeze(1)
        patches = z_pad.unfold(1, 2 * r + 1, 1)
        return self.net(patches).squeeze(-1)


class LatentSemigroupNetStencilV(nn.Module):
    """
    Latent semigroup learner with spatially-conditioned potential.

    Energy:
        Psi(z) = sum_i V_theta(stencil_i(z)) + 1/2 sum a_ij (z_i-z_j)^2

    Dynamics:
        dz/dt = -K(z) * grad Psi(z)

    grad Psi computed via autograd since V depends on multiple z components.
    """

    def __init__(self, N=64, m=0.0, M=1.0,
                 stencil_radius_V=2,
                 hidden_V=[64, 64], hidden_K=[64, 64],
                 stencil_radius_K=3, interaction_radius=2,
                 beta_V=0.0):
        super().__init__()
        self.N = N
        self.m = m
        self.M = M

        # Stencil-based potential (key difference from original)
        self.V_net = StencilVNet(radius=stencil_radius_V,
                                  hidden_dims=hidden_V, beta=beta_V)

        # Interaction coefficients (same as original)
        self.emb = nn.Parameter(torch.randn(N, 8) * 0.001)

        # Local interaction mask
        mask = torch.zeros(N, N)
        for i in range(N):
            for j in range(N):
                dist = min(abs(i - j), N - abs(i - j))
                if dist <= interaction_radius and i != j:
                    mask[i, j] = 1.0
        self.register_buffer('interaction_mask', mask)

        # Mobility network
        self.K_net = StencilMLP(radius=stencil_radius_K, hidden_dims=hidden_K)
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
        """Psi(z) = sum V(stencil(z)) + 1/2 sum a_ij (z_i-z_j)^2"""
        V_sum = self.V_net(z).sum(dim=1)  # (B,)

        a_full = F.softplus(torch.mm(self.emb, self.emb.t()))
        a_ij = a_full * self.interaction_mask
        diff = z.unsqueeze(2) - z.unsqueeze(1)
        interaction = 0.5 * (a_ij * diff.pow(2)).sum(dim=(1, 2))

        return V_sum + interaction

    def grad_psi(self, z, a_ij=None):
        """
        grad_z Psi(z) via autograd.

        Since V now depends on multiple z components (through stencil),
        we use autograd.grad instead of analytical computation.

        Args:
            z: (B, N) — must require_grad=True for autograd
            a_ij: (N, N) pre-computed interaction coefficients
        """
        if a_ij is None:
            a_full = F.softplus(torch.mm(self.emb, self.emb.t()))
            a_ij = a_full * self.interaction_mask

        # V gradient via autograd
        psi_V = self.V_net(z).sum(dim=1)  # (B,)
        grad_V = torch.autograd.grad(
            psi_V, z, grad_outputs=torch.ones_like(psi_V),
            create_graph=True, retain_graph=True
        )[0]  # (B, N)

        # Interaction gradient (analytical, same as original)
        diff = z.unsqueeze(2) - z.unsqueeze(1)
        grad_interaction = 2.0 * (a_ij * diff).sum(dim=2)

        return grad_V + grad_interaction

    def latent_dynamics(self, z, a_ij=None):
        """dz/dt = -K(z) * grad Psi(z)."""
        if a_ij is None:
            a_full = F.softplus(torch.mm(self.emb, self.emb.t()))
            a_ij = a_full * self.interaction_mask

        k = F.softplus(self.K_net(z)) + 5e-3
        g = self.grad_psi(z, a_ij=a_ij)
        return -k * g

    def rk4_step(self, z, dt, a_ij=None):
        """Single RK4 step. Recomputes grad_psi at each substep."""
        f1 = self.latent_dynamics(z, a_ij)
        f2 = self.latent_dynamics(z + 0.5 * dt * f1, a_ij)
        f3 = self.latent_dynamics(z + 0.5 * dt * f2, a_ij)
        f4 = self.latent_dynamics(z + dt * f3, a_ij)
        z_next = z + dt / 6.0 * (f1 + 2.0 * f2 + 2.0 * f3 + f4)
        return torch.clamp(z_next, -20.0, 20.0)

    def forward(self, u, tau):
        """Phi(u, tau): evolve u by time tau."""
        z = self.encode(u)

        # Cache interaction coefficients
        a_full = F.softplus(torch.mm(self.emb, self.emb.t()))
        a_ij = a_full * self.interaction_mask

        dt = tau / self.ode_steps
        with torch.enable_grad():
            z.requires_grad_(True)
            for _ in range(self.ode_steps):
                z = self.rk4_step(z, dt, a_ij)
                z.requires_grad_(True)
        return self.decode(z)

    def rollout(self, u0, tau, n_steps):
        u = u0
        traj = [u0]
        for _ in range(n_steps):
            u = self.forward(u, tau)
            traj.append(u)
        return torch.stack(traj, dim=0)
