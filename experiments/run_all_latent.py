#!/usr/bin/env python3
"""
Re-run all latent model experiments with fixed grad_psi.
BaselineResNet and FNO checkpoints are still valid (no grad_psi dependency).
Only retrains: LatentSemigroupNet (Fisher-KPP), LatentSemigroupNetBounded (Burgers, Allen-Cahn),
Ablation variants, FNO comparison latent, Error growth latent.
"""
import sys, os, time, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from models import LatentSemigroupNet, LatentSemigroupNetBounded, BaselineResNet, FNOBaseline
from training import train_model
from evaluate import evaluate_full

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
t_global = time.time()


# ============================================================
# Helper: generate PDE data (if not cached)
# ============================================================

def generate_fisher_kpp_data(N=64, L=10.0, nu=0.1, r=1.0, dt_pde=0.005,
                              n_train=1000, n_val=50, tau=0.1, T_max=2.0,
                              seed=42, device="cpu"):
    """Generate Fisher-KPP training data using central FD + RK4."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dx = L / N
    x = np.linspace(dx/2, L - dx/2, N, dtype=np.float64)

    # Laplacian stencil (periodic)
    def laplacian(u):
        return (np.roll(u, -1) + np.roll(u, 1) - 2*u) / dx**2

    def rhs(u):
        return nu * laplacian(u) + r * u * (1 - u)

    def rk4_step(u, dt):
        k1 = rhs(u)
        k2 = rhs(u + 0.5*dt*k1)
        k3 = rhs(u + 0.5*dt*k2)
        k4 = rhs(u + dt*k3)
        return u + dt/6*(k1 + 2*k2 + 2*k3 + k4)

    def solve_trajectory(u0, T, dt):
        steps = int(round(T / dt))
        traj = [u0.copy()]
        u = u0.copy()
        for _ in range(steps):
            u = rk4_step(u, dt)
            u = np.clip(u, 0, 1)
            traj.append(u.copy())
        return traj

    n_steps_per_tau = int(round(tau / dt_pde))
    n_snapshots = int(round(T_max / tau))

    # Training data
    train_u0_list, train_ut_list = [], []
    for _ in range(n_train):
        u0 = 0.3 + 0.4 * np.random.rand(N)
        traj = solve_trajectory(u0, T_max, dt_pde)
        for s in range(n_snapshots):
            idx0 = s * n_steps_per_tau
            idx1 = (s + 1) * n_steps_per_tau
            if idx1 < len(traj):
                train_u0_list.append(traj[idx0])
                train_ut_list.append(traj[idx1])

    train_u0 = torch.tensor(np.array(train_u0_list), dtype=torch.float32)
    train_ut = torch.tensor(np.array(train_ut_list), dtype=torch.float32)

    # Validation trajectories
    val_trajs = []
    val_u0_list = []
    for _ in range(n_val):
        u0 = 0.3 + 0.4 * np.random.rand(N)
        traj = solve_trajectory(u0, T_max, dt_pde)
        val_u0_list.append(traj[0])
        t_true = torch.arange(len(traj)) * dt_pde
        u_true = torch.tensor(np.array(traj), dtype=torch.float32)
        val_trajs.append((t_true, u_true))

    val_u0 = torch.tensor(np.array(val_u0_list), dtype=torch.float32)

    return train_u0, train_ut, val_u0, val_trajs


def generate_burgers_data(N=64, L=2.0, nu=0.01, dt_pde=0.001,
                           n_train=1000, n_val=50, tau=0.05, T_max=1.0,
                           seed=42):
    """Generate viscous Burgers data (periodic BC, central FD + RK4)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dx = L / N
    x = np.linspace(dx/2, L - dx/2, N, dtype=np.float64)

    def rhs(u):
        lap = (np.roll(u, -1) + np.roll(u, 1) - 2*u) / dx**2
        # Upwind for convection
        du_conv = np.where(u >= 0,
                           u * (u - np.roll(u, 1)) / dx,
                           u * (np.roll(u, -1) - u) / dx)
        return nu * lap - du_conv

    def rk4_step(u, dt):
        k1 = rhs(u)
        k2 = rhs(u + 0.5*dt*k1)
        k3 = rhs(u + 0.5*dt*k2)
        k4 = rhs(u + dt*k3)
        return u + dt/6*(k1 + 2*k2 + 2*k3 + k4)

    def solve_trajectory(u0, T, dt):
        steps = int(round(T / dt))
        traj = [u0.copy()]
        u = u0.copy()
        for _ in range(steps):
            u = rk4_step(u, dt)
            u = np.clip(u, -1, 1)
            traj.append(u.copy())
        return traj

    n_steps_per_tau = int(round(tau / dt_pde))
    n_snapshots = int(round(T_max / tau))

    train_u0_list, train_ut_list = [], []
    for _ in range(n_train):
        u0 = np.sin(2*np.pi*x/L) * (0.5 + 0.5*np.random.rand())
        u0 += 0.1 * np.random.randn(N)
        u0 = np.clip(u0, -0.9, 0.9)
        traj = solve_trajectory(u0, T_max, dt_pde)
        for s in range(n_snapshots):
            idx0 = s * n_steps_per_tau
            idx1 = (s + 1) * n_steps_per_tau
            if idx1 < len(traj):
                train_u0_list.append(traj[idx0])
                train_ut_list.append(traj[idx1])

    train_u0 = torch.tensor(np.array(train_u0_list), dtype=torch.float32)
    train_ut = torch.tensor(np.array(train_ut_list), dtype=torch.float32)

    val_trajs = []
    val_u0_list = []
    for _ in range(n_val):
        u0 = np.sin(2*np.pi*x/L) * (0.5 + 0.5*np.random.rand())
        u0 += 0.1 * np.random.randn(N)
        u0 = np.clip(u0, -0.9, 0.9)
        traj = solve_trajectory(u0, T_max, dt_pde)
        val_u0_list.append(traj[0])
        t_true = torch.arange(len(traj)) * dt_pde
        u_true = torch.tensor(np.array(traj), dtype=torch.float32)
        val_trajs.append((t_true, u_true))

    val_u0 = torch.tensor(np.array(val_u0_list), dtype=torch.float32)
    return train_u0, train_ut, val_u0, val_trajs


def generate_allen_cahn_data(N=64, L=1.0, epsilon=0.1, dt_pde=0.0005,
                              n_train=1000, n_val=50, tau=0.1, T_max=2.0,
                              seed=42):
    """Generate Allen-Cahn data (periodic BC, central FD + IMEX)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dx = L / N
    x = np.linspace(dx/2, L - dx/2, N, dtype=np.float64)

    def laplacian(u):
        return (np.roll(u, -1) + np.roll(u, 1) - 2*u) / dx**2

    def rhs_nonlinear(u):
        return u - u**3

    def imex_step(u, dt):
        # Semi-implicit: linear diffusion implicit, nonlinear explicit
        # (I - dt*eps*Laplacian) u^{n+1} = u^n + dt*f(u^n)
        rhs_explicit = u + dt * rhs_nonlinear(u)
        # Solve tridiagonal system with periodic BC (use spectral method)
        u_hat = np.fft.fft(rhs_explicit)
        k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        u_hat = u_hat / (1 + dt * epsilon * k**2)
        return np.real(np.fft.ifft(u_hat))

    def solve_trajectory(u0, T, dt):
        steps = int(round(T / dt))
        traj = [u0.copy()]
        u = u0.copy()
        for _ in range(steps):
            u = imex_step(u, dt)
            u = np.clip(u, -1, 1)
            traj.append(u.copy())
        return traj

    n_steps_per_tau = int(round(tau / dt_pde))
    n_snapshots = int(round(T_max / tau))

    train_u0_list, train_ut_list = [], []
    for _ in range(n_train):
        u0 = np.tanh((x - 0.5) / (np.sqrt(2) * epsilon))
        u0 += 0.05 * np.random.randn(N)
        u0 = np.clip(u0, -0.95, 0.95)
        traj = solve_trajectory(u0, T_max, dt_pde)
        for s in range(n_snapshots):
            idx0 = s * n_steps_per_tau
            idx1 = (s + 1) * n_steps_per_tau
            if idx1 < len(traj):
                train_u0_list.append(traj[idx0])
                train_ut_list.append(traj[idx1])

    train_u0 = torch.tensor(np.array(train_u0_list), dtype=torch.float32)
    train_ut = torch.tensor(np.array(train_ut_list), dtype=torch.float32)

    val_trajs = []
    val_u0_list = []
    for _ in range(n_val):
        u0 = np.tanh((x - 0.5) / (np.sqrt(2) * epsilon))
        u0 += 0.05 * np.random.randn(N)
        u0 = np.clip(u0, -0.95, 0.95)
        traj = solve_trajectory(u0, T_max, dt_pde)
        val_u0_list.append(traj[0])
        t_true = torch.arange(len(traj)) * dt_pde
        u_true = torch.tensor(np.array(traj), dtype=torch.float32)
        val_trajs.append((t_true, u_true))

    val_u0 = torch.tensor(np.array(val_u0_list), dtype=torch.float32)
    return train_u0, train_ut, val_u0, val_trajs


# ============================================================
# Experiment 1: Fisher-KPP Latent
# ============================================================

def run_fisher_kpp_latent():
    print("\n" + "="*60)
    print("EXPERIMENT 1: Fisher-KPP Latent (N=64)")
    print("="*60)

    ckpt_dir = "checkpoints"
    data_path = os.path.join(ckpt_dir, "data.pt")
    results_dir = "results"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    N = 64
    tau = 0.1
    n_epochs = 100

    # Generate or load data
    if os.path.exists(data_path):
        print("Loading cached data...")
        data = torch.load(data_path, weights_only=False)
        train_u0, train_ut, val_u0, val_trajs = data["train_u0"], data["train_ut"], data["val_u0"], data["val_trajs"]
    else:
        print("Generating Fisher-KPP data...")
        train_u0, train_ut, val_u0, val_trajs = generate_fisher_kpp_data(N=N, tau=tau)
        torch.save({"train_u0": train_u0, "train_ut": train_ut,
                    "val_u0": val_u0, "val_trajs": val_trajs}, data_path)
        print(f"Data saved to {data_path}")

    print(f"Train: {train_u0.shape}, Val: {val_u0.shape}")

    # Train latent model
    model = LatentSemigroupNet(
        N=N, hidden_V=[64, 64], hidden_K=[64, 64],
        stencil_radius=3, beta_V=0.0
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LatentSemigroupNet params: {n_params}")

    t0 = time.time()
    history = train_model(
        model, train_u0, train_ut, val_u0, val_trajs,
        tau=tau, n_epochs=n_epochs, batch_size=64, lr=1e-3,
        alpha_rollout=0.0, alpha_energy=0.0, alpha_bound=0.1, alpha_V=0.0,
        weight_decay=1e-5, checkpoint_dir=ckpt_dir,
        model_name="latent", device=DEVICE
    )
    print(f"Fisher-KPP Latent training: {time.time()-t0:.0f}s")

    # Evaluate
    best_ckpt = torch.load(os.path.join(ckpt_dir, "latent_best.pt"),
                           map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    metrics, details = evaluate_full(
        model, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=DEVICE, model_name="Latent"
    )
    print(f"  MSE: {metrics['rollout_mse_mean']:.4e} ± {metrics['rollout_mse_std']:.4e}")
    print(f"  BV:  {metrics['bound_viol_mean']:.4e}")
    print(f"  SD:  {metrics['semigroup_defect_mean']:.4e}")

    return metrics, details


# ============================================================
# Experiment 2: Fisher-KPP Baseline + FNO (evaluate existing)
# ============================================================

def run_fisher_kpp_baselines():
    print("\n" + "="*60)
    print("EXPERIMENT 2: Fisher-KPP Baselines (evaluate existing)")
    print("="*60)

    ckpt_dir = "checkpoints"
    N = 64
    tau = 0.1

    data_path = os.path.join(ckpt_dir, "data.pt")
    data = torch.load(data_path, weights_only=False)
    val_u0, val_trajs = data["val_u0"], data["val_trajs"]

    results = {}

    # BaselineResNet
    baseline_path = os.path.join(ckpt_dir, "baseline_best.pt")
    if os.path.exists(baseline_path):
        print("Evaluating BaselineResNet...")
        baseline = BaselineResNet(N=N, hidden_dim=32, n_blocks=4)
        baseline = baseline.to(DEVICE)
        ckpt = torch.load(baseline_path, map_location=DEVICE, weights_only=False)
        baseline.load_state_dict(ckpt["model_state_dict"])
        m, _ = evaluate_full(baseline, val_u0, val_trajs, tau=tau,
                             rollout_steps=20, device=DEVICE, model_name="BaselineResNet",
                             lower_bound=0.0, upper_bound=1.0)
        results["baseline_resnet"] = m
        print(f"  MSE: {m['rollout_mse_mean']:.4e}")
    else:
        print("  WARNING: baseline_best.pt not found, skipping")

    # FNO
    fno_path = os.path.join(ckpt_dir, "fno", "fno_best.pt")
    if os.path.exists(fno_path):
        print("Evaluating FNO...")
        fno = FNOBaseline(N=N, width=32, n_modes=16, n_layers=4)
        fno = fno.to(DEVICE)
        ckpt = torch.load(fno_path, map_location=DEVICE, weights_only=False)
        fno.load_state_dict(ckpt["model_state_dict"])
        m, _ = evaluate_full(fno, val_u0, val_trajs, tau=tau,
                             rollout_steps=20, device=DEVICE, model_name="FNO",
                             lower_bound=0.0, upper_bound=1.0)
        results["fno"] = m
        print(f"  MSE: {m['rollout_mse_mean']:.4e}")
    else:
        print("  WARNING: fno_best.pt not found, skipping")

    return results


# ============================================================
# Experiment 3: Allen-Cahn Latent
# ============================================================

def run_allen_cahn_latent():
    print("\n" + "="*60)
    print("EXPERIMENT 3: Allen-Cahn Latent (N=64)")
    print("="*60)

    ckpt_dir = "checkpoints/allen_cahn"
    data_path = os.path.join(ckpt_dir, "data_N64.pt")
    os.makedirs(ckpt_dir, exist_ok=True)

    N = 64
    tau = 0.1
    n_epochs = 100

    if os.path.exists(data_path):
        print("Loading cached Allen-Cahn data...")
        data = torch.load(data_path, weights_only=False)
        train_u0, train_ut, val_u0, val_trajs = data["train_u0"], data["train_ut"], data["val_u0"], data["val_trajs"]
    else:
        print("Generating Allen-Cahn data...")
        train_u0, train_ut, val_u0, val_trajs = generate_allen_cahn_data(N=N, tau=tau)
        torch.save({"train_u0": train_u0, "train_ut": train_ut,
                    "val_u0": val_u0, "val_trajs": val_trajs}, data_path)

    print(f"Train: {train_u0.shape}, Val: {val_u0.shape}")

    model = LatentSemigroupNetBounded(
        N=N, m=-1.0, M=1.0, hidden_V=[64, 64], hidden_K=[64, 64],
        stencil_radius=3, beta_V=0.0
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LatentSemigroupNetBounded params: {n_params}")

    t0 = time.time()
    history = train_model(
        model, train_u0, train_ut, val_u0, val_trajs,
        tau=tau, n_epochs=n_epochs, batch_size=64, lr=1e-3,
        alpha_rollout=0.0, alpha_energy=0.0, alpha_bound=0.1, alpha_V=0.0,
        weight_decay=1e-5, checkpoint_dir=ckpt_dir,
        model_name="latent_N64", device=DEVICE,
        lower_bound=-1.0, upper_bound=1.0
    )
    print(f"Allen-Cahn Latent training: {time.time()-t0:.0f}s")

    best_ckpt = torch.load(os.path.join(ckpt_dir, "latent_N64_best.pt"),
                           map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    metrics, details = evaluate_full(
        model, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=DEVICE, model_name="Latent",
        lower_bound=-1.0, upper_bound=1.0
    )
    print(f"  MSE: {metrics['rollout_mse_mean']:.4e} ± {metrics['rollout_mse_std']:.4e}")
    print(f"  BV:  {metrics['bound_viol_mean']:.4e}")
    print(f"  SD:  {metrics['semigroup_defect_mean']:.4e}")

    return metrics, details


# ============================================================
# Experiment 4: Burgers Latent
# ============================================================

def run_burgers_latent():
    print("\n" + "="*60)
    print("EXPERIMENT 4: Burgers Latent (N=64)")
    print("="*60)

    ckpt_dir = "checkpoints/burgers"
    data_path = os.path.join(ckpt_dir, "data_N64.pt")
    os.makedirs(ckpt_dir, exist_ok=True)

    N = 64
    tau = 0.05
    n_epochs = 100

    if os.path.exists(data_path):
        print("Loading cached Burgers data...")
        data = torch.load(data_path, weights_only=False)
        train_u0, train_ut, val_u0, val_trajs = data["train_u0"], data["train_ut"], data["val_u0"], data["val_trajs"]
    else:
        print("Generating Burgers data...")
        train_u0, train_ut, val_u0, val_trajs = generate_burgers_data(N=N, tau=tau)
        torch.save({"train_u0": train_u0, "train_ut": train_ut,
                    "val_u0": val_u0, "val_trajs": val_trajs}, data_path)

    print(f"Train: {train_u0.shape}, Val: {val_u0.shape}")

    model = LatentSemigroupNetBounded(
        N=N, m=-1.0, M=1.0, hidden_V=[64, 64], hidden_K=[64, 64],
        stencil_radius=3, beta_V=0.0
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LatentSemigroupNetBounded params: {n_params}")

    t0 = time.time()
    history = train_model(
        model, train_u0, train_ut, val_u0, val_trajs,
        tau=tau, n_epochs=n_epochs, batch_size=64, lr=1e-3,
        alpha_rollout=0.0, alpha_energy=0.0, alpha_bound=0.1, alpha_V=0.0,
        weight_decay=1e-5, checkpoint_dir=ckpt_dir,
        model_name="latent_N64", device=DEVICE,
        lower_bound=-1.0, upper_bound=1.0
    )
    print(f"Burgers Latent training: {time.time()-t0:.0f}s")

    best_ckpt = torch.load(os.path.join(ckpt_dir, "latent_N64_best.pt"),
                           map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    metrics, details = evaluate_full(
        model, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=DEVICE, model_name="Latent",
        lower_bound=-1.0, upper_bound=1.0
    )
    print(f"  MSE: {metrics['rollout_mse_mean']:.4e} ± {metrics['rollout_mse_std']:.4e}")
    print(f"  BV:  {metrics['bound_viol_mean']:.4e}")
    print(f"  SD:  {metrics['semigroup_defect_mean']:.4e}")

    return metrics, details


# ============================================================
# Experiment 5: Ablation Study (latent variants only)
# ============================================================

class LatentNoInteraction(LatentSemigroupNet):
    """Ablation: no interaction term (a_ij = 0)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.emb.data.zero_()
        self.emb.requires_grad_(False)

    def psi(self, z):
        return self.V_net(z.unsqueeze(-1)).squeeze(-1).sum(dim=1)

    def grad_psi(self, z):
        with torch.enable_grad():
            zi = z.detach().requires_grad_(True)
            V_sum = self.V_net(zi.unsqueeze(-1)).squeeze(-1).sum()
            return torch.autograd.grad(V_sum, zi, create_graph=self.training)[0]


class LatentNoMobility(LatentSemigroupNet):
    """Ablation: constant mobility K = 1."""
    def K_diag(self, z):
        return torch.ones(z.shape, device=z.device)


class LatentNoInteractionBounded(LatentSemigroupNetBounded):
    """Ablation: no interaction term for bounded case."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.emb.data.zero_()
        self.emb.requires_grad_(False)

    def psi(self, z):
        return self.V_net(z.unsqueeze(-1)).squeeze(-1).sum(dim=1)

    def grad_psi(self, z):
        with torch.enable_grad():
            zi = z.detach().requires_grad_(True)
            V_sum = self.V_net(zi.unsqueeze(-1)).squeeze(-1).sum()
            return torch.autograd.grad(V_sum, zi, create_graph=self.training)[0]


class LatentNoMobilityBounded(LatentSemigroupNetBounded):
    """Ablation: constant mobility K = 1 for bounded case."""
    def K_diag(self, z):
        return torch.ones(z.shape, device=z.device)


def run_ablation():
    print("\n" + "="*60)
    print("EXPERIMENT 5: Ablation Study")
    print("="*60)

    ckpt_dir = "checkpoints/ablation"
    os.makedirs(ckpt_dir, exist_ok=True)

    N = 64
    tau = 0.1
    n_epochs = 100

    # Load Fisher-KPP data
    data = torch.load("checkpoints/data.pt", weights_only=False)
    train_u0, train_ut, val_u0, val_trajs = data["train_u0"], data["train_ut"], data["val_u0"], data["val_trajs"]

    results = {}

    variants = [
        ("full", LatentSemigroupNet, {}),
        ("no_interaction", LatentNoInteraction, {"interaction_radius": 0}),
        ("no_mobility", LatentNoMobility, {}),
    ]

    for name, ModelClass, extra_kwargs in variants:
        print(f"\n--- Ablation: {name} ---")
        kwargs = dict(N=N, hidden_V=[64, 64], hidden_K=[64, 64],
                      stencil_radius=3, beta_V=0.0)
        kwargs.update(extra_kwargs)
        model = ModelClass(**kwargs)

        train_model(
            model, train_u0, train_ut, val_u0, val_trajs,
            tau=tau, n_epochs=n_epochs, batch_size=64, lr=1e-3,
            alpha_rollout=0.0, alpha_energy=0.0,
            weight_decay=1e-5, checkpoint_dir=ckpt_dir,
            model_name=f"{name}", device=DEVICE
        )

        best_ckpt = torch.load(os.path.join(ckpt_dir, f"{name}_best.pt"),
                               map_location=DEVICE, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        metrics, _ = evaluate_full(
            model, val_u0, val_trajs, tau=tau,
            rollout_steps=20, device=DEVICE, model_name=name
        )
        results[name] = {
            "mse": metrics["rollout_mse_mean"],
            "mse_std": metrics["rollout_mse_std"],
            "bv": metrics["bound_viol_mean"]
        }
        print(f"  {name}: MSE={metrics['rollout_mse_mean']:.4e}, BV={metrics['bound_viol_mean']:.4e}")

    return results


# ============================================================
# Experiment 6: Error Growth (long-time rollout)
# ============================================================

def run_error_growth():
    print("\n" + "="*60)
    print("EXPERIMENT 6: Error Growth (long-time rollout)")
    print("="*60)

    N = 64
    tau = 0.1

    # Load data
    data = torch.load("checkpoints/data.pt", weights_only=False)
    val_u0_full, val_trajs_full = data["val_u0"], data["val_trajs"]
    # Use first 20 for long-time
    val_u0 = val_u0_full[:20]
    val_trajs = val_trajs_full[:20]

    results = {"metadata": {"N": N, "tau": tau, "n_test": 20, "T_max": 20.0}}

    # Latent (need to load best checkpoint)
    print("Evaluating Latent...")
    model = LatentSemigroupNet(N=N, hidden_V=[64, 64], hidden_K=[64, 64],
                                stencil_radius=3, beta_V=0.0)
    ckpt = torch.load("checkpoints/latent_best.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    metrics, details = evaluate_full(
        model, val_u0, val_trajs, tau=tau,
        rollout_steps=200, device=DEVICE, model_name="Latent"
    )
    results["latent"] = {**metrics, **details}
    print(f"  Latent MSE@T=20: {metrics['rollout_mse_mean']:.4e}")

    # BaselineResNet
    baseline_path = "checkpoints/baseline_best.pt"
    if os.path.exists(baseline_path):
        print("Evaluating BaselineResNet...")
        baseline = BaselineResNet(N=N, hidden_dim=32, n_blocks=4)
        baseline = baseline.to(DEVICE)
        ckpt = torch.load(baseline_path, map_location=DEVICE, weights_only=False)
        baseline.load_state_dict(ckpt["model_state_dict"])
        metrics, details = evaluate_full(
            baseline, val_u0, val_trajs, tau=tau,
            rollout_steps=200, device=DEVICE, model_name="BaselineResNet",
            lower_bound=0.0, upper_bound=1.0
        )
        results["baseline_resnet"] = {**metrics, **details}
        print(f"  Baseline MSE@T=20: {metrics['rollout_mse_mean']:.4e}")

    # FNO
    fno_path = "checkpoints/fno/fno_best.pt"
    if os.path.exists(fno_path):
        print("Evaluating FNO...")
        fno = FNOBaseline(N=N, width=32, n_modes=16, n_layers=4)
        fno = fno.to(DEVICE)
        ckpt = torch.load(fno_path, map_location=DEVICE, weights_only=False)
        fno.load_state_dict(ckpt["model_state_dict"])
        metrics, details = evaluate_full(
            fno, val_u0, val_trajs, tau=tau,
            rollout_steps=200, device=DEVICE, model_name="FNO",
            lower_bound=0.0, upper_bound=1.0
        )
        results["fno"] = {**metrics, **details}
        print(f"  FNO MSE@T=20: {metrics['rollout_mse_mean']:.4e}")

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("RE-RUN ALL LATENT EXPERIMENTS (fixed grad_psi)")
    print("="*60)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    all_results = {}

    # 1. Fisher-KPP Latent
    fkpp_latent_metrics, fkpp_latent_details = run_fisher_kpp_latent()
    all_results["fisher_kpp_latent"] = fkpp_latent_metrics

    # 2. Fisher-KPP Baselines (evaluate existing)
    fkpp_baseline_results = run_fisher_kpp_baselines()
    all_results.update(fkpp_baseline_results)

    # Save Fisher-KPP comparison JSON
    comparison = {
        "latent": fkpp_latent_metrics,
        "baseline": fkpp_baseline_results.get("baseline_resnet", {}),
        "fno": fkpp_baseline_results.get("fno", {})
    }
    with open("results/comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print("\nSaved results/comparison.json")

    # 3. Allen-Cahn Latent
    ac_metrics = run_allen_cahn_latent()
    all_results["allen_cahn_latent"] = ac_metrics

    # Evaluate Allen-Cahn baselines (existing checkpoints)
    print("\n--- Allen-Cahn Baselines ---")
    ac_baseline_results = {}
    for name, ModelClass, path, lb, ub in [
        ("baseline", BaselineResNet, "checkpoints/allen_cahn/baseline_N64_best.pt", -1.0, 1.0),
        ("fno", FNOBaseline, "checkpoints/allen_cahn/fno_N64_best.pt", -1.0, 1.0),
    ]:
        if os.path.exists(path):
            print(f"Evaluating Allen-Cahn {name}...")
            if name == "baseline":
                m = ModelClass(N=64, hidden_dim=32, n_blocks=4)
            else:
                m = ModelClass(N=64, width=32, n_modes=16, n_layers=4)
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            m.load_state_dict(ckpt["model_state_dict"])
            m = m.to(DEVICE)
            data = torch.load("checkpoints/allen_cahn/data_N64.pt", weights_only=False)
            metrics, _ = evaluate_full(m, data["val_u0"], data["val_trajs"], tau=0.1,
                                       rollout_steps=20, device=DEVICE, model_name=name,
                                       lower_bound=lb, upper_bound=ub)
            ac_baseline_results[name] = metrics
            print(f"  {name}: MSE={metrics['rollout_mse_mean']:.4e}")

    # Save Allen-Cahn comparison JSON
    ac_comparison = {
        "latent": ac_metrics,
        "fno": ac_baseline_results.get("fno", {}),
        "baseline_resnet": ac_baseline_results.get("baseline", {})
    }
    with open("results/allen_cahn_comparison.json", "w") as f:
        json.dump(ac_comparison, f, indent=2)
    print("Saved results/allen_cahn_comparison.json")

    # 4. Burgers Latent
    burgers_metrics = run_burgers_latent()
    all_results["burgers_latent"] = burgers_metrics

    # Evaluate Burgers baselines (existing checkpoints)
    print("\n--- Burgers Baselines ---")
    burgers_baseline_results = {}
    for name, ModelClass, path, lb, ub in [
        ("baseline_resnet", BaselineResNet, "checkpoints/burgers/baseline_resnet_best.pt", -1.0, 1.0),
        ("fno", FNOBaseline, "checkpoints/burgers/fno_best.pt", -1.0, 1.0),
    ]:
        if os.path.exists(path):
            print(f"Evaluating Burgers {name}...")
            if "resnet" in name:
                m = ModelClass(N=64, hidden_dim=32, n_blocks=4)
            else:
                m = ModelClass(N=64, width=32, n_modes=16, n_layers=4)
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            m.load_state_dict(ckpt["model_state_dict"])
            m = m.to(DEVICE)
            data = torch.load("checkpoints/burgers/data_N64.pt", weights_only=False)
            metrics, _ = evaluate_full(m, data["val_u0"], data["val_trajs"], tau=0.05,
                                       rollout_steps=20, device=DEVICE, model_name=name,
                                       lower_bound=-1.0, upper_bound=1.0)
            burgers_baseline_results[name] = metrics
            print(f"  {name}: MSE={metrics['rollout_mse_mean']:.4e}")

    # Save Burgers comparison JSON
    burgers_comparison = {
        "latent": burgers_metrics,
        "fno": burgers_baseline_results.get("fno", {}),
        "baseline_resnet": burgers_baseline_results.get("baseline_resnet", {})
    }
    with open("results/burgers_comparison.json", "w") as f:
        json.dump(burgers_comparison, f, indent=2)
    print("Saved results/burgers_comparison.json")

    # 5. Ablation
    ablation_results = run_ablation()
    with open("results/ablation.json", "w") as f:
        json.dump(ablation_results, f, indent=2)
    print("Saved results/ablation.json")

    # 6. Error Growth
    error_growth_results = run_error_growth()
    with open("results/error_growth.json", "w") as f:
        json.dump(error_growth_results, f, indent=2, default=str)
    print("Saved results/error_growth.json")

    # Summary
    total_time = time.time() - t_global
    print("\n" + "="*60)
    print(f"ALL EXPERIMENTS COMPLETE ({total_time:.0f}s = {total_time/3600:.1f}h)")
    print("="*60)
    print("\nResults files:")
    for f in ["comparison.json", "allen_cahn_comparison.json", "burgers_comparison.json",
              "ablation.json", "error_growth.json"]:
        path = f"results/{f}"
        if os.path.exists(path):
            print(f"  ✓ {path}")
