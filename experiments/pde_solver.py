"""
High-accuracy PDE reference solver for 1D Fisher-KPP equation.

    d_t u = nu * d_xx u + r * u * (1-u)

on [0, L] with periodic boundary conditions.

Uses spectral (Fourier) method with ETD-RK4 (exponential time
differencing + Runge-Kutta 4).
"""
import torch
import numpy as np


class FisherKPPSolver:
    """Spectral ETD-RK4 solver for 1D Fisher-KPP with periodic BC."""

    def __init__(
        self,
        N=64,
        L=10.0,
        nu=0.1,
        r=1.0,
        dt=0.01,
        dtype=torch.float32,
    ):
        if int(N) <= 1:
            raise ValueError("N must be greater than one")
        if not np.isfinite(dt) or float(dt) <= 0:
            raise ValueError("dt must be a positive finite number")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("FisherKPPSolver supports float32 or float64")
        self.N = N
        self.L = L
        self.nu = nu
        self.r = r
        self.dt = dt
        self.dtype = dtype
        self.dx = L / N

        # Fourier wavenumbers (RFFT: N//2+1 for even N)
        k_rfft = 2.0 * np.pi * np.fft.rfftfreq(N, d=self.dx)
        self.k = torch.tensor(k_rfft, dtype=dtype)
        self.k2 = self.k ** 2

        # Linear operator in Fourier space (diffusion)
        self._L = -self.nu * self.k2

        # Integrating factors
        self._E = torch.exp(self._L * dt)
        self._E2 = torch.exp(self._L * dt / 2.0)

    def _N_hat(self, u_phys):
        """Nonlinear reaction term: FFT(r * u * (1-u))."""
        return self.r * torch.fft.rfft(u_phys * (1.0 - u_phys))

    def step(self, u_hat):
        """ETD-RK4 single time step."""
        dt = self.dt
        E = self._E
        E2 = self._E2

        u_phys = torch.fft.irfft(u_hat, n=self.N)

        # Stage 1
        N1 = self._N_hat(u_phys)
        a_hat = E2 * u_hat + dt / 2.0 * E2 * N1
        a_phys = torch.fft.irfft(a_hat, n=self.N)

        # Stage 2
        N2 = self._N_hat(a_phys)
        b_hat = E2 * u_hat + dt / 2.0 * N2
        b_phys = torch.fft.irfft(b_hat, n=self.N)

        # Stage 3
        N3 = self._N_hat(b_phys)
        c_hat = E2 * a_hat + dt / 2.0 * (2.0 * N3 - N1)
        c_phys = torch.fft.irfft(c_hat, n=self.N)

        # Stage 4
        N4 = self._N_hat(c_phys)

        u_next = E * u_hat
        u_next = u_next + dt / 6.0 * (
            E * N1 + 2.0 * E2 * (N2 + N3) + N4
        )
        return u_next

    def solve(self, u0, T, save_every=1):
        """
        Solve from initial condition u0 to time T.

        Args:
            u0: tensor (N,) initial condition in physical space
            T: float, final time
            save_every: int, save every N steps (1 = every step)

        Returns:
            t: tensor (n_save,)
            u: tensor (n_save, N)
        """
        ratio = float(T) / float(self.dt)
        n_steps = int(round(ratio))
        if not np.isclose(ratio, n_steps, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "T must be an integer multiple of dt; "
                f"received T/dt={ratio:.17g}"
            )
        if int(save_every) <= 0:
            raise ValueError("save_every must be positive")
        save_every = int(save_every)
        if n_steps == 0:
            return torch.tensor([0.0], dtype=self.dtype), u0.to(self.dtype).unsqueeze(0)

        n_save = n_steps // save_every + 1

        t_vals = torch.zeros(n_save, dtype=self.dtype)
        u_vals = torch.zeros(n_save, self.N, dtype=self.dtype)

        u_phys = u0.to(self.dtype).clone()
        u_hat = torch.fft.rfft(u_phys)

        t_vals[0] = 0.0
        u_vals[0] = u_phys.clone()
        save_idx = 1

        for step in range(1, n_steps + 1):
            u_hat = self.step(u_hat)
            if step % save_every == 0:
                u_phys = torch.fft.irfft(u_hat, n=self.N)
                t_vals[save_idx] = step * self.dt
                u_vals[save_idx] = u_phys.clone()
                save_idx += 1

        return t_vals, u_vals


def generate_initial_conditions(N, n_samples, L=10.0):
    """
    Generate random smooth initial conditions for Fisher-KPP.

    Uses superposition of low-frequency Fourier modes.
    """
    x = torch.linspace(0, L, N + 1)[:N]
    u0_list = []

    for _ in range(n_samples):
        n_modes = np.random.randint(2, 6)
        u0 = torch.zeros(N)
        for _ in range(n_modes):
            k = np.random.randint(1, 4)
            phase = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(0.05, 0.3)
            u0 = u0 + amp * torch.sin(2 * np.pi * k * x / L + phase)
        u0 = 0.4 + 0.4 * u0 / (u0.abs().max() + 1e-8)
        u0 = u0.clamp(0.05, 0.95)
        u0_list.append(u0)

    return torch.stack(u0_list)


def generate_training_data(
    N=64, n_train=500, n_val=50, L=10.0, nu=0.1, r=1.0,
    dt=0.005, T_max=2.0, tau=0.1
):
    """
    Generate training data: (u0, u_tau) pairs and full trajectories.

    Returns:
        train_u0, train_ut: tensor (n_train, N), (n_train, N)
        val_u0: tensor (n_val, N)
        val_trajs: list of (t, u) for validation
    """
    solver = FisherKPPSolver(N=N, L=L, nu=nu, r=r, dt=dt)
    n_tau_steps = int(tau / dt)

    print(f"Generating {n_train} training initial conditions...")
    train_u0 = generate_initial_conditions(N, n_train, L)
    train_ut = torch.zeros_like(train_u0)

    for i in range(n_train):
        t, u = solver.solve(train_u0[i], tau, save_every=n_tau_steps)
        train_ut[i] = u[-1]
        if (i + 1) % 100 == 0:
            print(f"  train: {i+1}/{n_train}")

    print(f"Generating {n_val} validation trajectories...")
    val_u0 = generate_initial_conditions(N, n_val, L)
    val_trajs = []

    for i in range(n_val):
        t, u = solver.solve(val_u0[i], T_max, save_every=1)
        val_trajs.append((t, u))
        if (i + 1) % 10 == 0:
            print(f"  val: {i+1}/{n_val}")

    return train_u0, train_ut, val_u0, val_trajs


if __name__ == "__main__":
    N = 64
    solver = FisherKPPSolver(N=N)
    u0 = generate_initial_conditions(N, 1)[0]
    t, u = solver.solve(u0, T=2.0, save_every=10)
    print(f"  trajectory: {t.shape[0]} snapshots")
    print(f"  u range: [{u.min():.4f}, {u.max():.4f}]")


# ============================================================
# 2D Fisher-KPP Equation Solver
# ============================================================

class FisherKPP2DSolver:
    """Spectral RK4 solver for 2D Fisher-KPP equation with periodic BC.

        du/dt = nu * Laplacian(u) + r * u * (1-u),  (x,y) in [0,L]x[0,L]

    Uses FFT2-based spectral differentiation for spatial derivatives
    and explicit RK4 for time integration.
    """

    def __init__(self, Nx=32, Ny=32, L=2*np.pi, nu=0.1, r=1.0, dt=0.005):
        self.Nx = Nx
        self.Ny = Ny
        self.L = L
        self.nu = nu
        self.r = r
        self.dt = dt
        self.dx = L / Nx
        self.dy = L / Ny
        self.N_total = Nx * Ny

        # 2D Fourier wavenumbers
        kx_fft = 2.0 * np.pi * np.fft.fftfreq(Nx, d=self.dx)
        ky_fft = 2.0 * np.pi * np.fft.fftfreq(Ny, d=self.dy)
        KX, KY = np.meshgrid(kx_fft, ky_fft, indexing='ij')
        self.K2 = torch.tensor(KX**2 + KY**2, dtype=torch.float64)

    def _rhs(self, u_hat):
        """Right-hand side in Fourier space."""
        u = torch.fft.ifft2(u_hat).real
        # Diffusion: nu * Laplacian(u) -> -nu * k^2 * u_hat
        diffusion = -self.nu * self.K2 * u_hat
        # Reaction: r * u * (1-u)
        reaction = torch.fft.fft2(self.r * u * (1.0 - u))
        return diffusion + reaction

    def rk4_step(self, u_hat):
        """Single RK4 step in Fourier space."""
        dt = self.dt
        k1 = self._rhs(u_hat)
        k2 = self._rhs(u_hat + 0.5 * dt * k1)
        k3 = self._rhs(u_hat + 0.5 * dt * k2)
        k4 = self._rhs(u_hat + dt * k3)
        return u_hat + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def solve(self, u0, T, save_every=1):
        """
        Solve from initial condition u0 to time T.

        Args:
            u0: tensor (Nx, Ny) initial condition in physical space
            T: float, final time
            save_every: int, save every N steps (1 = every step)

        Returns:
            t: tensor (n_save,)
            u: tensor (n_save, Nx*Ny)  (flattened)
        """
        n_steps = int(T / self.dt)
        if n_steps == 0:
            return torch.tensor([0.0]), u0.reshape(1, -1)

        n_save = n_steps // save_every + 1
        t_vals = torch.zeros(n_save)
        u_vals = torch.zeros(n_save, self.N_total)

        u_hat = torch.fft.fft2(u0.double())
        t_vals[0] = 0.0
        u_vals[0] = u0.reshape(-1).clone()
        save_idx = 1

        for step in range(1, n_steps + 1):
            u_hat = self.rk4_step(u_hat)
            if step % save_every == 0:
                u_phys = torch.fft.ifft2(u_hat).real
                t_vals[save_idx] = step * self.dt
                u_vals[save_idx] = u_phys.reshape(-1).float().clone()
                save_idx += 1

        return t_vals, u_vals


def generate_fisher_kpp_2d_initial_conditions(n_samples, Nx=32, Ny=32, L=2*np.pi):
    """
    Generate random smooth initial conditions for 2D Fisher-KPP.

    Uses superposition of low-frequency 2D Fourier modes.
    Values clamped to (0, 1).
    """
    x = torch.linspace(0, L, Nx + 1)[:Nx]
    y = torch.linspace(0, L, Ny + 1)[:Ny]
    XX, YY = torch.meshgrid(x, y, indexing='ij')

    u0_list = []
    for _ in range(n_samples):
        n_modes = np.random.randint(2, 6)
        u0 = torch.zeros(Nx, Ny)
        for _ in range(n_modes):
            kx = np.random.randint(1, 4)
            ky = np.random.randint(1, 4)
            phase_x = np.random.uniform(0, 2 * np.pi)
            phase_y = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(0.05, 0.3)
            u0 = u0 + amp * (
                torch.sin(2 * np.pi * kx * XX / L + phase_x) *
                torch.sin(2 * np.pi * ky * YY / L + phase_y)
            )
        u0 = 0.4 + 0.4 * u0 / (u0.abs().max() + 1e-8)
        u0 = u0.clamp(0.05, 0.95)
        u0_list.append(u0.reshape(-1))  # flatten to (Nx*Ny,)

    return torch.stack(u0_list)  # (n_samples, Nx*Ny)


def generate_fisher_kpp_2d_training_data(
    n_train=500, n_val=50, Nx=32, Ny=32, L=2*np.pi,
    nu=0.1, r=1.0, dt=0.005, T_max=2.0, tau=0.1
):
    """
    Generate training data for 2D Fisher-KPP equation.

    Returns:
        train_u0, train_ut: tensor (n_train, Nx*Ny), (n_train, Nx*Ny)
        val_u0: tensor (n_val, Nx*Ny)
        val_trajs: list of (t, u) for validation
    """
    solver = FisherKPP2DSolver(Nx=Nx, Ny=Ny, L=L, nu=nu, r=r, dt=dt)
    n_tau_steps = int(tau / dt)

    print(f"Generating {n_train} 2D Fisher-KPP training initial conditions...")
    train_u0 = generate_fisher_kpp_2d_initial_conditions(n_train, Nx, Ny, L)
    train_ut = torch.zeros_like(train_u0)

    for i in range(n_train):
        u0_2d = train_u0[i].reshape(Nx, Ny)
        t, u = solver.solve(u0_2d, tau, save_every=n_tau_steps)
        train_ut[i] = u[-1]
        if (i + 1) % 100 == 0:
            print(f"  train: {i+1}/{n_train}")

    print(f"Generating {n_val} 2D Fisher-KPP validation trajectories...")
    val_u0 = generate_fisher_kpp_2d_initial_conditions(n_val, Nx, Ny, L)
    val_trajs = []

    for i in range(n_val):
        u0_2d = val_u0[i].reshape(Nx, Ny)
        t, u = solver.solve(u0_2d, T_max, save_every=n_tau_steps)
        val_trajs.append((t, u))
        if (i + 1) % 10 == 0:
            print(f"  val: {i+1}/{n_val}")

    return train_u0, train_ut, val_u0, val_trajs


# ============================================================
# Viscous Burgers Equation Solver
# ============================================================

class BurgersSolver:
    """Spectral RK4 solver for 1D viscous Burgers equation with periodic BC.

        du/dt + u * du/dx = nu * d2u/dx2,  x in [0, L]

    Uses FFT-based spectral differentiation for spatial derivatives
    and explicit RK4 for time integration.
    """

    def __init__(self, N=64, L=2*np.pi, nu=0.01, dt=0.001):
        self.N = N
        self.L = L
        self.nu = nu
        self.dt = dt
        self.dx = L / N

        # Fourier wavenumbers
        k_fft = 2.0 * np.pi * np.fft.fftfreq(N, d=self.dx)
        self.k = torch.tensor(k_fft, dtype=torch.float64)
        self.k2 = self.k ** 2

    def _rhs(self, u_hat):
        """Right-hand side in Fourier space."""
        # u in physical space
        u = torch.fft.ifft(u_hat).real
        # Nonlinear term: -u * du/dx = -0.5 * d(u^2)/dx
        # In Fourier: -0.5 * ik * FFT(u^2)
        u_sq_hat = torch.fft.fft(u ** 2)
        nonlinear = -0.5 * self.k * 1j * u_sq_hat
        # Linear diffusion: -nu * k^2 * u_hat
        diffusion = -self.nu * self.k2 * u_hat
        return nonlinear + diffusion

    def rk4_step(self, u_hat):
        """Single RK4 step in Fourier space."""
        dt = self.dt
        k1 = self._rhs(u_hat)
        k2 = self._rhs(u_hat + 0.5 * dt * k1)
        k3 = self._rhs(u_hat + 0.5 * dt * k2)
        k4 = self._rhs(u_hat + dt * k3)
        return u_hat + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def solve(self, u0, T, save_every=1):
        """
        Solve from initial condition u0 to time T.

        Args:
            u0: tensor (N,) initial condition in physical space
            T: float, final time
            save_every: int, save every N steps

        Returns:
            t: tensor (n_save,)
            u: tensor (n_save, N)
        """
        n_steps = int(T / self.dt)
        if n_steps == 0:
            return torch.tensor([0.0]), u0.unsqueeze(0)

        n_save = n_steps // save_every + 1
        t_vals = torch.zeros(n_save)
        u_vals = torch.zeros(n_save, self.N)

        u_hat = torch.fft.fft(u0.double())
        t_vals[0] = 0.0
        u_vals[0] = u0.clone()
        save_idx = 1

        for step in range(1, n_steps + 1):
            u_hat = self.rk4_step(u_hat)
            if step % save_every == 0:
                u_phys = torch.fft.ifft(u_hat).real
                t_vals[save_idx] = step * self.dt
                u_vals[save_idx] = u_phys.float().clone()
                save_idx += 1

        return t_vals, u_vals

    def energy(self, u):
        """E(u) = 0.5 * integral u^2 dx."""
        return 0.5 * (u ** 2).sum(dim=-1) * self.dx


def generate_burgers_initial_conditions(N, n_samples, L=2*np.pi):
    """Generate random smooth initial conditions for Burgers.
    Values in [-1, 1], zero mean oscillatory."""
    x = torch.linspace(0, L, N + 1)[:N]
    u0_list = []

    for _ in range(n_samples):
        n_modes = np.random.randint(2, 6)
        u0 = torch.zeros(N)
        for _ in range(n_modes):
            k = np.random.randint(1, 4)
            phase = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(0.1, 0.5)
            u0 = u0 + amp * torch.sin(2 * np.pi * k * x / L + phase)
        u0 = u0 / (u0.abs().max() + 1e-8) * 0.9
        u0 = u0.clamp(-1.0, 1.0)
        u0_list.append(u0)

    return torch.stack(u0_list)


def generate_burgers_training_data(
    N=64, n_train=1000, n_val=50, L=2*np.pi, nu=0.01,
    dt=0.001, T_max=1.0, tau=0.05
):
    """Generate training data for Burgers equation."""
    solver = BurgersSolver(N=N, L=L, nu=nu, dt=dt)
    n_tau_steps = int(tau / dt)

    print(f"Generating {n_train} Burgers training initial conditions...")
    train_u0 = generate_burgers_initial_conditions(N, n_train, L)
    train_ut = torch.zeros_like(train_u0)

    for i in range(n_train):
        t, u = solver.solve(train_u0[i], tau, save_every=n_tau_steps)
        train_ut[i] = u[-1]
        if (i + 1) % 200 == 0:
            print(f"  train: {i+1}/{n_train}")

    print(f"Generating {n_val} Burgers validation trajectories...")
    val_u0 = generate_burgers_initial_conditions(N, n_val, L)
    val_trajs = []

    for i in range(n_val):
        t, u = solver.solve(val_u0[i], T_max, save_every=n_tau_steps)
        val_trajs.append((t, u))
        if (i + 1) % 10 == 0:
            print(f"  val: {i+1}/{n_val}")

    return train_u0, train_ut, val_u0, val_trajs


# ============================================================
# Allen-Cahn Equation Solver
# ============================================================

class AllenCahnSolver:
    """Spectral RK4 solver for 1D Allen-Cahn equation with periodic BC.

        du/dt = eps^2 * d2u/dx2 + u(1 - u^2),  x in [0, L]

    Uses FFT-based spectral differentiation for spatial derivatives
    and explicit RK4 for time integration.
    """

    def __init__(self, N=64, L=2*np.pi, eps=0.1, dt=0.001):
        self.N = N
        self.L = L
        self.eps = eps
        self.dt = dt
        self.dx = L / N

        # Fourier wavenumbers
        k_fft = 2.0 * np.pi * np.fft.fftfreq(N, d=self.dx)
        self.k = torch.tensor(k_fft, dtype=torch.float64)
        self.k2 = self.k ** 2

    def _rhs(self, u_hat):
        """Right-hand side in Fourier space."""
        u = torch.fft.ifft(u_hat).real
        # Linear diffusion: eps^2 * d2u/dx2 -> -eps^2 * k^2 * u_hat
        diffusion = -self.eps ** 2 * self.k2 * u_hat
        # Nonlinear reaction: u(1 - u^2) = u - u^3
        nonlinear = torch.fft.fft(u * (1.0 - u ** 2))
        return diffusion + nonlinear

    def rk4_step(self, u_hat):
        """Single RK4 step in Fourier space."""
        dt = self.dt
        k1 = self._rhs(u_hat)
        k2 = self._rhs(u_hat + 0.5 * dt * k1)
        k3 = self._rhs(u_hat + 0.5 * dt * k2)
        k4 = self._rhs(u_hat + dt * k3)
        return u_hat + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def solve(self, u0, T, save_every=1):
        """
        Solve from initial condition u0 to time T.

        Args:
            u0: tensor (N,) initial condition in physical space
            T: float, final time
            save_every: int, save every N steps

        Returns:
            t: tensor (n_save,)
            u: tensor (n_save, N)
        """
        n_steps = int(T / self.dt)
        if n_steps == 0:
            return torch.tensor([0.0]), u0.unsqueeze(0)

        n_save = n_steps // save_every + 1
        t_vals = torch.zeros(n_save)
        u_vals = torch.zeros(n_save, self.N)

        u_hat = torch.fft.fft(u0.double())
        t_vals[0] = 0.0
        u_vals[0] = u0.clone()
        save_idx = 1

        for step in range(1, n_steps + 1):
            u_hat = self.rk4_step(u_hat)
            if step % save_every == 0:
                u_phys = torch.fft.ifft(u_hat).real
                t_vals[save_idx] = step * self.dt
                u_vals[save_idx] = u_phys.float().clone()
                save_idx += 1

        return t_vals, u_vals

    def energy(self, u):
        """Allen-Cahn energy: E(u) = integral(eps^2/2 |grad u|^2 + (1-u^2)^2/4) dx.

        Computed via FFT (spectral derivative).
        """
        u_hat = torch.fft.fft(u.double())
        # |grad u|^2 in Fourier: sum k^2 |u_hat|^2 / N^2
        grad_sq = (self.k2 * u_hat.abs() ** 2).sum() / self.N ** 2
        # (1-u^2)^2/4 in physical space
        potential = ((1.0 - u ** 2) ** 2 / 4.0).sum() * self.dx
        return (0.5 * self.eps ** 2 * grad_sq + potential).float()


def generate_allen_cahn_initial_conditions(N, n_samples, L=2*np.pi):
    """Generate random smooth initial conditions for Allen-Cahn.
    Values in (-1, 1), using superposition of low-frequency Fourier modes.
    """
    x = torch.linspace(0, L, N + 1)[:N]
    u0_list = []

    for _ in range(n_samples):
        n_modes = np.random.randint(2, 6)
        u0 = torch.zeros(N)
        for _ in range(n_modes):
            k = np.random.randint(1, 4)
            phase = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(0.1, 0.5)
            u0 = u0 + amp * torch.sin(2 * np.pi * k * x / L + phase)
        # Normalize and clamp to (-1, 1)
        u0 = u0 / (u0.abs().max() + 1e-8) * 0.9
        u0 = u0.clamp(-0.95, 0.95)
        u0_list.append(u0)

    return torch.stack(u0_list)


def generate_allen_cahn_training_data(
    N=64, n_train=1000, n_val=50, L=2*np.pi, eps=0.1,
    dt=0.001, T_max=2.0, tau=0.1
):
    """Generate training data for Allen-Cahn equation.

    Returns:
        train_u0, train_ut: tensor (n_train, N), (n_train, N)
        val_u0: tensor (n_val, N)
        val_trajs: list of (t, u) for validation
    """
    solver = AllenCahnSolver(N=N, L=L, eps=eps, dt=dt)
    n_tau_steps = int(tau / dt)

    print(f"Generating {n_train} Allen-Cahn training initial conditions...")
    train_u0 = generate_allen_cahn_initial_conditions(N, n_train, L)
    train_ut = torch.zeros_like(train_u0)

    for i in range(n_train):
        t, u = solver.solve(train_u0[i], tau, save_every=n_tau_steps)
        train_ut[i] = u[-1]
        if (i + 1) % 200 == 0:
            print(f"  train: {i+1}/{n_train}")

    print(f"Generating {n_val} Allen-Cahn validation trajectories...")
    val_u0 = generate_allen_cahn_initial_conditions(N, n_val, L)
    val_trajs = []

    for i in range(n_val):
        t, u = solver.solve(val_u0[i], T_max, save_every=n_tau_steps)
        val_trajs.append((t, u))
        if (i + 1) % 10 == 0:
            print(f"  val: {i+1}/{n_val}")

    return train_u0, train_ut, val_u0, val_trajs
