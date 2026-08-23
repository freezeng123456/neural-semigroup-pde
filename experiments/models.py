"""
Structure-preserving latent semigroup learner and baseline models.

Based on: semigroup_bound_dissipative_one_step_network_note.tex

LatentSemigroupNet:
    u = sigmoid(z),  z' = -K_theta(z) * grad Psi_theta(z)
    Architecture-level guarantees: K_N=[0,1]^N invariant, energy dissipates.

BaselineResNet:
    Standard ResNet one-step predictor (no structure guarantees).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utility layers
# ============================================================

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
        """x: (..., 1) -> (..., 1)"""
        return self.net(x) + 0.5 * self.beta.abs() * x.pow(2)

    def value_and_grad(self, x):
        """Compute V(x) and dV/dx analytically in one forward pass.

        Avoids autograd.grad which creates a Hessian bottleneck that
        attenuates V_net gradients by ~10^6 during backprop through
        ODE integration steps.

        Args:
            x: (..., 1) input
        Returns:
            val: (..., 1) V(x)
            grad: (..., 1) dV/dx
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, 1)  # (B, 1)
        h = x_flat  # (B, cur_dim)
        dh = torch.ones(x_flat.shape[0], 1, device=x.device)  # (B, 1): dV/dh

        linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]

        for i, layer in enumerate(linear_layers):
            W = layer.weight  # (out_dim, in_dim)
            pre = layer(h)    # (B, out_dim)
            if i < len(linear_layers) - 1:
                # Hidden layer: Linear -> Softplus
                sig = torch.sigmoid(pre)        # (B, out_dim)
                dh = dh @ W.T * sig             # (B, out_dim): chain rule
                h = F.softplus(pre)             # (B, out_dim)
            else:
                # Output layer: Linear only
                dh = dh @ W.T                   # (B, 1)
                h = pre

        # Quadratic term: 0.5 * beta * x^2 => derivative = beta * x
        beta = self.beta.abs()
        val = h + 0.5 * beta * x_flat.pow(2)
        grad = dh + beta * x_flat

        return val.reshape(orig_shape), grad.reshape(orig_shape)


class StencilMLP(nn.Module):
    """Stencil-based MLP for spatial interactions (like 1D convolution)."""
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
        """z: (batch, N) -> (batch, N) with zero-padding"""
        B, N = z.shape
        r = self.radius
        # Pad
        z_pad = F.pad(z.unsqueeze(1), (r, r), mode='constant', value=0.0).squeeze(1)
        # Extract patches: (B, N, 2r+1)
        patches = z_pad.unfold(1, 2*r+1, 1)
        # MLP per site
        out = self.net(patches).squeeze(-1)  # (B, N)
        return out


# ============================================================
# Latent Semigroup Network
# ============================================================

class LatentSemigroupNet(nn.Module):
    """
    Structure-preserving latent semigroup learner.

    Architecture (from paper Sections 4-5):
        u = sigmoid(z)                                    (latent map)
        dz/dt = -K_theta(z) * grad Psi_theta(z)           (latent ODE)
        Psi_theta(z) = sum V(z_i) + 1/2 sum a_ij (z_i - z_j)^2
        K_theta(z) = diag(softplus(k_i(z)))

    Guarantees by construction:
        - u_i in (0,1) always
        - d/dt E_theta(u) <= 0 along latent flow
    """

    def __init__(self, N=64, hidden_V=[32, 32], hidden_K=[32, 32],
                 stencil_radius=3, beta_V=0.0, interaction_radius=2):
        super().__init__()
        self.N = N

        # Scalar potential V: R -> R
        self.V_net = ScalarMLP(hidden_V, beta=beta_V)

        # Interaction coefficients a_ij via learned embeddings
        self.emb = nn.Parameter(torch.randn(N, 8) * 0.001)

        # Local interaction mask: only connect |i-j| <= radius (periodic)
        mask = torch.zeros(N, N)
        for i in range(N):
            for j in range(N):
                dist = min(abs(i - j), N - abs(i - j))
                if dist <= interaction_radius and i != j:
                    mask[i, j] = 1.0
        self.register_buffer('interaction_mask', mask)

        # Mobility K_theta: stencil MLP -> softplus
        self.K_net = StencilMLP(radius=stencil_radius, hidden_dims=hidden_K)

        # rk4 steps for ODE integration
        self.ode_steps = 30  # rk4 steps per tau
        self.register_buffer('eps', torch.tensor(1e-6))

        # Initialize weights very small for ODE stability at start
        self._init_weights()

    def _init_weights(self):
        """Very-small-weight initialization for ODE stability at init."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, u):
        """u in (eps, 1-eps) -> z = log(u/(1-u))"""
        u_clamped = u.clamp(self.eps, 1.0 - self.eps)
        return torch.log(u_clamped / (1.0 - u_clamped))

    def decode(self, z):
        """z -> u = sigmoid(z)"""
        return torch.sigmoid(z)

    def energy(self, u):
        """E_theta(u) = Psi_theta(g^{-1}(u)) — induced discrete energy."""
        z = self.encode(u)
        return self.psi(z)

    def psi(self, z):
        """Psi_theta(z) = sum V(z_i) + 1/2 sum a_ij (z_i - z_j)^2."""
        # V term
        V_sum = self.V_net(z.unsqueeze(-1)).squeeze(-1).sum(dim=1)  # (B,)

        # Interaction term with local mask
        a_full = F.softplus(torch.mm(self.emb, self.emb.t()))  # (N, N)
        a_ij = a_full * self.interaction_mask  # (N, N), local only
        diff = z.unsqueeze(2) - z.unsqueeze(1)  # (B, N, N)
        interaction = 0.5 * (a_ij * diff.pow(2)).sum(dim=(1, 2))  # (B,)

        return V_sum + interaction

    def grad_psi(self, z, a_ij=None):
        """grad_z Psi_theta(z).

        Uses analytical V'(z) computation to avoid the Hessian bottleneck
        that autograd.grad introduces (10^6x gradient attenuation).

        Args:
            z: (B, N) latent state
            a_ij: (N, N) pre-computed interaction coefficients (optional).
                  If None, computed fresh (e.g. for standalone calls from psi/energy).
        """
        # V gradient: analytical, avoids autograd.grad Hessian bottleneck
        _V_val, dV_dz = self.V_net.value_and_grad(z.unsqueeze(-1))
        grad_V = dV_dz.squeeze(-1)  # (B, N)

        # Interaction gradient (analytical)
        # d/dz_k [1/2 sum_{i,j} a_ij (z_i-z_j)^2] = 2 sum_j a_kj (z_k-z_j)
        if a_ij is None:
            a_full = F.softplus(torch.mm(self.emb, self.emb.t()))
            a_ij = a_full * self.interaction_mask
        diff = z.unsqueeze(2) - z.unsqueeze(1)
        grad_interaction = 2.0 * (a_ij * diff).sum(dim=2)

        return grad_V + grad_interaction

    def _dynamics_and_grad(self, z, a_ij):
        """Compute K_diag and grad_psi in one call, sharing the input z.

        Args:
            z: (B, N) latent state
            a_ij: (N, N) pre-computed interaction coefficients

        Returns:
            K: (B, N) diagonal mobility coefficients
            grad_Psi: (B, N) gradient of Psi w.r.t. z
        """
        K = F.softplus(self.K_net(z)) + 5e-3
        grad_Psi = self.grad_psi(z, a_ij=a_ij)
        return K, grad_Psi

    def latent_dynamics(self, z, a_ij=None):
        """dz/dt = -K_theta(z) * grad Psi_theta(z)."""
        if a_ij is None:
            a_full = F.softplus(torch.mm(self.emb, self.emb.t()))
            a_ij = a_full * self.interaction_mask
        k, g = self._dynamics_and_grad(z, a_ij)
        return -k * g

    def rk4_step(self, z, dt, a_ij=None):
        """Single RK4 step of latent ODE."""
        f1 = self.latent_dynamics(z, a_ij)
        f2 = self.latent_dynamics(z + 0.5 * dt * f1, a_ij)
        f3 = self.latent_dynamics(z + 0.5 * dt * f2, a_ij)
        f4 = self.latent_dynamics(z + dt * f3, a_ij)
        z_next = z + dt / 6.0 * (f1 + 2.0 * f2 + 2.0 * f3 + f4)
        # Clip extreme z values for numerical stability
        return torch.clamp(z_next, -20.0, 20.0)

    def forward(self, u, tau):
        """Phi_theta(u, tau): evolve u by time tau."""
        z = self.encode(u)
        # Compute a_ij once per forward pass (120x savings)
        a_full = F.softplus(torch.mm(self.emb, self.emb.t()))
        a_ij = a_full * self.interaction_mask
        dt = tau / self.ode_steps
        for _ in range(self.ode_steps):
            z = self.rk4_step(z, dt, a_ij)
        return self.decode(z)

    def latent_V_values(self, u, tau):
        """Collect V(z_i) values along the latent trajectory.

        Provides direct gradient path to V_net via L_V = mean(V^2),
        bypassing the Hessian bottleneck. Collects V(z) at the start
        of each RK4 step (30 values total).

        Returns:
            V_loss: scalar, mean of V(z_i)^2 over trajectory
        """
        z = self.encode(u)
        dt = tau / self.ode_steps
        V_squares = []
        for _ in range(self.ode_steps):
            V_val = self.V_net(z.unsqueeze(-1)).squeeze(-1)  # (B, N)
            V_squares.append(V_val.pow(2).mean())
            z = self.rk4_step(z, dt)
        return torch.stack(V_squares).mean()

    def rollout(self, u0, tau, n_steps):
        """Multi-step rollout: u0 -> u1 -> u2 -> ... -> u_n."""
        u = u0
        traj = [u0]
        for _ in range(n_steps):
            u = self.forward(u, tau)
            traj.append(u)
        return torch.stack(traj, dim=0)  # (n_steps+1, B, N)


# ============================================================
# Baseline ResNet predictor
# ============================================================

class ResBlock1D(nn.Module):
    """1D residual block with conv."""
    def __init__(self, channels=32, kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.act = nn.Softplus()

    def forward(self, x):
        # x: (B, C, N)
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(x + h)


class BaselineResNet(nn.Module):
    """
    Standard ResNet one-step predictor.
    No structural guarantees.
    """

    def __init__(self, N=64, hidden_dim=32, n_blocks=3):
        super().__init__()
        self.N = N
        self.enc = nn.Conv1d(1, hidden_dim, 5, padding=2)
        self.blocks = nn.Sequential(*[ResBlock1D(hidden_dim) for _ in range(n_blocks)])
        self.dec = nn.Conv1d(hidden_dim, 1, 5, padding=2)

    def forward(self, u, tau):
        """u: (B, N) -> (B, N) prediction at time u+tau."""
        B = u.shape[0]
        # Encode
        h = self.enc(u.unsqueeze(1))  # (B, C, N)
        h = self.blocks(h)
        # Decode
        du = self.dec(h).squeeze(1)  # (B, N)
        return u + du


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":
    N = 64
    B = 4
    u = 0.3 + 0.4 * torch.rand(B, N)
    tau = torch.tensor(0.1)

    print("=== LatentSemigroupNet ===")
    model = LatentSemigroupNet(N=N)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params}")
    out = model(u, tau)
    print(f"  Input range: [{u.min():.3f}, {u.max():.3f}]")
    print(f"  Output range: [{out.min():.3f}, {out.max():.3f}]")
    E_before = model.energy(u)
    E_after = model.energy(out)
    print(f"  Energy before: {E_before.mean():.4f}")
    print(f"  Energy after:  {E_after.mean():.4f}")
    print(f"  Energy monotone: {(E_after <= E_before + 1e-3).all()}")

    print("\n=== BaselineResNet ===")
    model2 = BaselineResNet(N=N)
    n_params2 = sum(p.numel() for p in model2.parameters())
    print(f"  Parameters: {n_params2}")
    out2 = model2(u, tau)
    print(f"  Output range: [{out2.min():.3f}, {out2.max():.3f}]")


# ============================================================
# FNO Baseline (Fourier Neural Operator)
# ============================================================

class SpectralConv1d(nn.Module):
    """1D Spectral convolution layer."""
    def __init__(self, in_channels, out_channels, n_modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, n_modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        # x: (B, C, N)
        B = x.shape[0]
        x_ft = torch.fft.rfft(x)  # (B, C, N//2+1)

        # Multiply relevant modes
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)
        n_modes = min(self.n_modes, x_ft.shape[-1])
        out_ft[:, :, :n_modes] = torch.einsum("bix,iox->box", x_ft[:, :, :n_modes], self.weights[:, :, :n_modes])

        return torch.fft.irfft(out_ft, n=x.shape[-1])


class FNO1DBlock(nn.Module):
    """Single FNO block: spectral conv + local conv + residual."""
    def __init__(self, width, n_modes):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, n_modes)
        self.local = nn.Conv1d(width, width, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.spectral(x) + self.local(x))


class FNOBaseline(nn.Module):
    """
    1D Fourier Neural Operator baseline.
    No structure preservation guarantees.
    """
    def __init__(self, N=64, width=32, n_modes=16, n_layers=4):
        super().__init__()
        self.N = N
        # Lift: (B, 1, N) -> (B, width, N)
        self.lift = nn.Conv1d(1, width, 1)
        self.blocks = nn.Sequential(*[FNO1DBlock(width, n_modes) for _ in range(n_layers)])
        # Project: (B, width, N) -> (B, 1, N)
        self.project = nn.Conv1d(width, 1, 1)

    def forward(self, u, tau):
        """u: (B, N) -> (B, N). tau is ignored (one-step map)."""
        x = u.unsqueeze(1)  # (B, 1, N)
        x = self.lift(x)
        x = self.blocks(x)
        out = self.project(x).squeeze(1)  # (B, N)
        return out


# ============================================================
# Latent Semigroup Net for [m, M] (Burgers: [-1, 1])
# ============================================================

class LatentSemigroupNetBounded(nn.Module):
    """
    Variant of LatentSemigroupNet for admissible set [m, M]^N.
    Uses u = m + (M-m)*sigmoid(z).
    """

    def __init__(self, N=64, m=-1.0, M=1.0, hidden_V=[32, 32], hidden_K=[32, 32],
                 stencil_radius=3, beta_V=0.0, interaction_radius=2):
        super().__init__()
        self.N = N
        self.m = m
        self.M = M

        self.V_net = ScalarMLP(hidden_V, beta=beta_V)
        self.emb = nn.Parameter(torch.randn(N, 8) * 0.001)

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
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0.0, std=0.001)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

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
        V_sum = self.V_net(z.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        a_full = F.softplus(torch.mm(self.emb, self.emb.t()))
        a_ij = a_full * self.interaction_mask
        diff = z.unsqueeze(2) - z.unsqueeze(1)
        interaction = 0.5 * (a_ij * diff.pow(2)).sum(dim=(1, 2))
        return V_sum + interaction

    def grad_psi(self, z, a_ij=None):
        """grad_z Psi_theta(z). Analytical V'(z) — no Hessian bottleneck."""
        _V_val, dV_dz = self.V_net.value_and_grad(z.unsqueeze(-1))
        grad_V = dV_dz.squeeze(-1)  # (B, N)

        if a_ij is None:
            a_full = F.softplus(torch.mm(self.emb, self.emb.t()))
            a_ij = a_full * self.interaction_mask
        diff = z.unsqueeze(2) - z.unsqueeze(1)
        grad_interaction = 2.0 * (a_ij * diff).sum(dim=2)

        return grad_V + grad_interaction

    def _dynamics_and_grad(self, z, a_ij):
        """Compute K_diag and grad_psi in one call, sharing the input z."""
        K = F.softplus(self.K_net(z)) + 5e-3
        grad_Psi = self.grad_psi(z, a_ij=a_ij)
        return K, grad_Psi

    def latent_dynamics(self, z, a_ij=None):
        if a_ij is None:
            a_full = F.softplus(torch.mm(self.emb, self.emb.t()))
            a_ij = a_full * self.interaction_mask
        k, g = self._dynamics_and_grad(z, a_ij)
        return -k * g

    def rk4_step(self, z, dt, a_ij=None):
        f1 = self.latent_dynamics(z, a_ij)
        f2 = self.latent_dynamics(z + 0.5 * dt * f1, a_ij)
        f3 = self.latent_dynamics(z + 0.5 * dt * f2, a_ij)
        f4 = self.latent_dynamics(z + dt * f3, a_ij)
        z_next = z + dt / 6.0 * (f1 + 2.0 * f2 + 2.0 * f3 + f4)
        return torch.clamp(z_next, -20.0, 20.0)

    def forward(self, u, tau):
        z = self.encode(u)
        a_full = F.softplus(torch.mm(self.emb, self.emb.t()))
        a_ij = a_full * self.interaction_mask
        dt = tau / self.ode_steps
        for _ in range(self.ode_steps):
            z = self.rk4_step(z, dt, a_ij)
        return self.decode(z)

    def latent_V_values(self, u, tau):
        """Collect V(z_i)^2 along trajectory for auxiliary loss."""
        z = self.encode(u)
        dt = tau / self.ode_steps
        V_squares = []
        for _ in range(self.ode_steps):
            V_val = self.V_net(z.unsqueeze(-1)).squeeze(-1)
            V_squares.append(V_val.pow(2).mean())
            z = self.rk4_step(z, dt)
        return torch.stack(V_squares).mean()

    def rollout(self, u0, tau, n_steps):
        u = u0
        traj = [u0]
        for _ in range(n_steps):
            u = self.forward(u, tau)
            traj.append(u)
        return torch.stack(traj, dim=0)
