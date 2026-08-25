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


def _broadcast_positive_tau(tau, u):
    """Return ``tau`` as a positive ``(B, 1, N)`` conditioning channel.

    A scalar tau is shared by the batch. A one-dimensional tensor must have
    exactly one value per input sample. The returned channel remains connected
    to tau's autograd graph.
    """
    if u.ndim != 2:
        raise ValueError(f"u must have shape (B, N), got {tuple(u.shape)}")

    batch_size, n_sites = u.shape
    try:
        tau_tensor = torch.as_tensor(tau, device=u.device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(
            "tau must be a scalar or a tensor with shape (B,)"
        ) from exc

    if tau_tensor.is_complex():
        raise ValueError("tau must be a real-valued scalar or tensor")

    if tau_tensor.ndim == 0:
        tau_batch = tau_tensor.expand(batch_size)
    elif tau_tensor.ndim == 1 and tau_tensor.shape[0] == batch_size:
        tau_batch = tau_tensor
    else:
        raise ValueError(
            f"tau must be a scalar or have shape ({batch_size},), "
            f"got {tuple(tau_tensor.shape)}"
        )

    tau_batch = tau_batch.to(dtype=u.dtype)
    valid = torch.isfinite(tau_batch) & (tau_batch > 0)
    if not bool(valid.all().item()):
        raise ValueError("tau must contain only finite, strictly positive values")

    return tau_batch.reshape(batch_size, 1, 1).expand(-1, 1, n_sites)


# ============================================================
# Utility layers
# ============================================================

class ScalarMLP(nn.Module):
    """MLP mapping R -> R with an optional fixed coercive quadratic floor.

    ``beta`` remains the trainable coefficient used by archived checkpoints.
    ``beta_floor`` is deliberately a plain, non-trainable float so adding it
    does not change the state-dict schema.  A positive floor therefore gives
    a configuration-level coercivity guarantee without invalidating existing
    checkpoints.
    """
    def __init__(self, hidden_dims=[32, 32], beta=0.0, beta_floor=0.0):
        super().__init__()
        if beta_floor < 0:
            raise ValueError(f"beta_floor must be non-negative, got {beta_floor}")
        dims = [1] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)
        self.beta = nn.Parameter(torch.tensor(beta))
        self.beta_floor = float(beta_floor)

    @property
    def effective_beta(self):
        """Non-negative quadratic coefficient used by the potential."""
        return self.beta.abs() + self.beta_floor

    def forward(self, x):
        """x: (..., 1) -> (..., 1)"""
        return self.net(x) + 0.5 * self.effective_beta * x.pow(2)

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
        beta = self.effective_beta
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
        """z: (batch, N) -> (batch, N) with periodic padding.

        The PDE solvers and the interaction mask use periodic boundaries.  The
        mobility stencil must use the same topology so a cyclic spatial shift
        of an input produces the corresponding shift of the output.
        """
        B, N = z.shape
        r = self.radius
        # Periodically wrap neighbouring sites at the two domain boundaries.
        z_pad = F.pad(z.unsqueeze(1), (r, r), mode='circular').squeeze(1)
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
                 stencil_radius=3, beta_V=0.0, beta_V_floor=0.0,
                 interaction_radius=2):
        super().__init__()
        self.N = N

        # Scalar potential V: R -> R
        self.V_net = ScalarMLP(
            hidden_V, beta=beta_V, beta_floor=beta_V_floor
        )

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
        dt = _broadcast_positive_tau(tau, z)[:, 0, :1] / self.ode_steps
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
        dt = _broadcast_positive_tau(tau, z)[:, 0, :1] / self.ode_steps
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
# Time-conditioned baseline predictors
# ============================================================

class TimeConditionedResNet(nn.Module):
    """1D ResNet baseline conditioned on a positive evolution time.

    The input state and tau are concatenated as two channels: the first is
    the spatial state and the second is tau broadcast uniformly over space.
    The default ``width=18, blocks=3`` configuration is a small baseline
    suitable for parameter-count comparisons with the latent model.

    Args:
        N: Number of spatial sites.  The convolutional layers also support
            other spatial lengths at inference time.
        width: Number of hidden convolutional channels.
        blocks: Number of residual blocks.
        kernel_size: Odd 1D convolution kernel size.
        hidden_dim: Optional alias for ``width`` matching the legacy naming.
        n_blocks: Optional alias for ``blocks`` matching the legacy naming.
    """

    def __init__(self, N=64, width=18, blocks=3, kernel_size=5,
                 *, hidden_dim=None, n_blocks=None):
        super().__init__()
        if hidden_dim is not None:
            width = hidden_dim
        if n_blocks is not None:
            blocks = n_blocks
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if blocks < 0:
            raise ValueError(f"blocks must be non-negative, got {blocks}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )

        self.N = N
        self.width = width
        self.blocks_count = blocks
        self.enc = nn.Conv1d(2, width, kernel_size, padding=kernel_size // 2)
        self.blocks = nn.Sequential(
            *[ResBlock1D(width, kernel_size) for _ in range(blocks)]
        )
        self.dec = nn.Conv1d(width, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, u, tau):
        """Map ``u`` at positive time ``tau`` to a tensor of shape ``(B, N)``."""
        tau_channel = _broadcast_positive_tau(tau, u)
        x = torch.cat((u.unsqueeze(1), tau_channel), dim=1)
        h = self.enc(x)
        h = self.blocks(h)
        du = self.dec(h).squeeze(1)
        return u + du


class TimeConditionedFNO(nn.Module):
    """1D Fourier Neural Operator baseline conditioned on a positive tau.

    As in :class:`TimeConditionedResNet`, tau is supplied as a constant
    spatial channel.  ``modes``/``layers`` are the concise names for the
    Fourier-mode and block counts; ``n_modes``/``n_layers`` are accepted as
    aliases so existing FNO-style call sites can opt into this new class
    without changing their naming convention.
    """

    def __init__(self, N=64, width=16, modes=8, layers=4,
                 *, n_modes=None, n_layers=None):
        super().__init__()
        if n_modes is not None:
            modes = n_modes
        if n_layers is not None:
            layers = n_layers
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if modes <= 0:
            raise ValueError(f"modes must be positive, got {modes}")
        if layers < 0:
            raise ValueError(f"layers must be non-negative, got {layers}")

        self.N = N
        self.width = width
        self.modes = modes
        self.layers = layers
        # Lift (B, 2, N) -> (B, width, N); channels are u and broadcast tau.
        self.lift = nn.Conv1d(2, width, 1)
        self.blocks = nn.Sequential(
            *[FNO1DBlock(width, modes) for _ in range(layers)]
        )
        self.project = nn.Conv1d(width, 1, 1)

    def forward(self, u, tau):
        """Map ``u`` at positive time ``tau`` to a tensor of shape ``(B, N)``."""
        tau_channel = _broadcast_positive_tau(tau, u)
        x = torch.cat((u.unsqueeze(1), tau_channel), dim=1)
        x = self.lift(x)
        x = self.blocks(x)
        return self.project(x).squeeze(1)


# ============================================================
# Latent Semigroup Net for [m, M] (Burgers: [-1, 1])
# ============================================================

class LatentSemigroupNetBounded(nn.Module):
    """
    Variant of LatentSemigroupNet for admissible set [m, M]^N.
    Uses u = m + (M-m)*sigmoid(z).
    """

    def __init__(self, N=64, m=-1.0, M=1.0, hidden_V=[32, 32], hidden_K=[32, 32],
                 stencil_radius=3, beta_V=0.0, beta_V_floor=0.0,
                 interaction_radius=2):
        super().__init__()
        self.N = N
        self.m = m
        self.M = M

        self.V_net = ScalarMLP(
            hidden_V, beta=beta_V, beta_floor=beta_V_floor
        )
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
        dt = _broadcast_positive_tau(tau, z)[:, 0, :1] / self.ode_steps
        for _ in range(self.ode_steps):
            z = self.rk4_step(z, dt, a_ij)
        return self.decode(z)

    def latent_V_values(self, u, tau):
        """Collect V(z_i)^2 along trajectory for auxiliary loss."""
        z = self.encode(u)
        dt = _broadcast_positive_tau(tau, z)[:, 0, :1] / self.ode_steps
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


class DecodedInteractionLatentSemigroupNetBounded(LatentSemigroupNetBounded):
    """Bounded latent gradient flow with interactions in physical coordinates.

    The standard bounded latent model penalizes ``(z_i - z_j)^2``.  For
    Allen--Cahn, however, the gradient portion of the physical free energy is
    naturally expressed through differences of the bounded state ``u``.  This
    variant retains the same decoder, diagonal positive mobility, learned
    scalar potential, RK4 flow, and therefore the same admissibility,
    learned-energy dissipation, and autonomous-semigroup guarantees.  It only
    changes the interaction energy to

    ``1/2 sum_ij a_ij (decode(z_i) - decode(z_j))^2``.

    The interaction gradient is computed analytically so this structural
    ablation does not reintroduce an autograd/Hessian bottleneck.
    """

    def psi(self, z):
        V_sum = self.V_net(z.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        a_full = F.softplus(torch.mm(self.emb, self.emb.t()))
        a_ij = a_full * self.interaction_mask
        u = self.decode(z)
        diff = u.unsqueeze(2) - u.unsqueeze(1)
        interaction = 0.5 * (a_ij * diff.pow(2)).sum(dim=(1, 2))
        return V_sum + interaction

    def grad_psi(self, z, a_ij=None):
        """Analytical ``grad_z Psi`` for the decoded-state interaction."""
        _V_val, dV_dz = self.V_net.value_and_grad(z.unsqueeze(-1))
        grad_V = dV_dz.squeeze(-1)
        if a_ij is None:
            a_full = F.softplus(torch.mm(self.emb, self.emb.t()))
            a_ij = a_full * self.interaction_mask

        sigmoid_z = torch.sigmoid(z)
        u = self.m + (self.M - self.m) * sigmoid_z
        du_dz = (self.M - self.m) * sigmoid_z * (1.0 - sigmoid_z)
        diff = u.unsqueeze(2) - u.unsqueeze(1)
        # a_ij is symmetric because it is generated as emb @ emb.T.  The
        # derivative of 1/2 sum_ij a_ij (u_i-u_j)^2 therefore has factor 2.
        grad_interaction = 2.0 * du_dz * (a_ij * diff).sum(dim=2)
        return grad_V + grad_interaction


class DecodedInteractionJacobianMobilityLatentSemigroupNetBounded(
    DecodedInteractionLatentSemigroupNetBounded
):
    """Decoded-interaction flow with a physical-coordinate mobility pullback.

    Let ``u = decode(z)`` and write the energy in physical state coordinates.
    Then ``grad_z Psi = (du/dz) grad_u E``.  A physical-state gradient flow
    ``u_dot = -K_u grad_u E`` is represented in latent coordinates by
    ``z_dot = -K_u (du/dz)^(-2) grad_z Psi``.  The standard positive stencil
    mobility has to learn this state-dependent Jacobian factor indirectly.
    This variant supplies it explicitly and leaves the learnable, positive
    local physical mobility unchanged.  Clamping prevents numerical overflow
    at the sigmoid's asymptotes while retaining positive mobility and learned
    energy dissipation.
    """

    def _dynamics_and_grad(self, z, a_ij):
        base_mobility = F.softplus(self.K_net(z)) + 5e-3
        sigmoid_z = torch.sigmoid(z)
        du_dz = (self.M - self.m) * sigmoid_z * (1.0 - sigmoid_z)
        jacobian_sq = du_dz.pow(2).clamp_min(float(self.eps))
        K = base_mobility / jacobian_sq
        grad_Psi = self.grad_psi(z, a_ij=a_ij)
        return K, grad_Psi
