# Review and Fix Report: Neural Semigroup Project

## 1. Current research idea

The project aims to learn a continuous-time PDE evolution operator

\[
\Phi_\theta(u,\tau) \approx \mathcal S_\tau(u)
\]

rather than a simple one-step predictor. The intended structure is a latent gradient-flow dynamical system:

\[
\dot z=-K_\theta(z)\nabla\Psi_\theta(z),\qquad u=g(z).
\]

The main motivations are:

1. Preserve maximum-principle type bounds through bounded decoding.
2. Obtain dissipative dynamics through a Lyapunov-like latent energy.
3. Generate a family of time operators from one autonomous ODE, giving semigroup structure in the continuous model.

## 2. What is currently valid

- The bounded decoder idea is mathematically meaningful.
- The autonomous latent ODE provides a natural source of continuous-time composition.
- The gradient-flow parameterization gives monotonicity of the learned latent energy.

## 3. Main theoretical issues

### 3.1 Semigroup claim

The exact semigroup property belongs to the continuous ODE flow:

\[
\varphi^{t+s}=\varphi^s\circ\varphi^t.
\]

The RK4 implementation is only an approximation. Claims should distinguish continuous theory from numerical implementation.

### 3.2 Energy interpretation

The model guarantees decay of the learned energy:

\[
\Psi_\theta(z(t)).
\]

It does not automatically guarantee decay of the physical PDE energy.

### 3.3 Approximation theory

The implemented potential and mobility classes are restricted. General approximation claims for arbitrary gradient flows require weakening.

### 3.4 Burgers equation

Viscous Burgers contains transport dynamics and is not purely a gradient flow. It should be treated as an out-of-class empirical benchmark.

## 4. Experimental bug identified

The previous Fisher-KPP evaluation compared mismatched time points:

- PDE snapshots were stored every `dt=0.005`.
- Model rollout used `tau=0.1`.
- Evaluation compared prediction at `t+0.1` with truth at `t+0.005`.

This has been fixed in `experiments/evaluate.py` by adding `reference_dt` and matching the correct trajectory index.

## 5. Recommended next steps

1. Re-run all experiments with corrected time alignment.
2. Report semigroup defect as numerical approximate consistency, not exact equality.
3. Separate learned-energy monotonicity from physical energy preservation.
4. Narrow theoretical statements to the implemented model class.
5. Add ablations without auxiliary semigroup/energy losses if claiming architecture-only guarantees.
