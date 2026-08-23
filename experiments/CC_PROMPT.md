Read the paper at:
  /home/shuixinf/projects/DC-PINNs_LaTeX_arxiv_2604.13723/semigroup_bound_dissipative_one_step_network_note.tex

Then design and write numerical experiment code in:
  /home/shuixinf/projects/DC-PINNs_LaTeX_arxiv_2604.13723/experiments/

The paper proposes a structure-preserving latent semigroup learner for nonlinear PDEs, with a Fisher-KPP reaction-diffusion benchmark:
  \partial_t u = \nu \Delta u + r u(1-u),  u(0)=u_0

Key architecture (Section 4-5):
- Latent transform: u = g(z) where g = sigmoid (maps R^N → [0,1]^N)
- Latent ODE: dz/dt = -K_\theta(z) \nabla \Psi_\theta(z)
- K_\theta(z) symmetric positive semidefinite
- Energy: E_\theta(u) = \Psi_\theta(g^{-1}(u))
- By construction: K_N = [0,1]^N invariant, energy dissipates along latent flow

Training (Section 6):
- Semigroup-based objectives: one-step + multi-step rollout + energy regularization
- Match: Dg(z) K_\theta(z) \nabla \Psi_\theta(z) ≈ -(\nu \Delta_h u + r u \odot (1-u))

Your task — write complete, runnable Python experiment code:

1. pde_solver.py — High-accuracy PDE reference solver (1D Fisher-KPP, spectral or fine finite difference)
2. models.py — Two models:
   a) LatentSemigroupNet: the paper's structure-preserving architecture
      - Encoder: g^{-1} (inverse sigmoid, clamping (eps,1-eps))
      - K_theta: NN(z) * I or low-rank PSD parameterization
      - Psi_theta: scalar NN, convex ICNN-style
      - Decoder: g (sigmoid)
      - Forward: ODE solve dz/dt = -K_theta(z) grad Psi_theta(z), then u = sigmoid(z)
   b) BaselineResNet: standard ResNet one-step predictor (no structure)
3. training.py — Training loop:
   - L_step: one-step prediction loss (MSE)
   - L_rollout: multi-step rollout loss (MSE)
   - L_energy: energy monotonicity violation penalty
   - L_reg: weight regularization
   - Combined: L = L_step + α L_rollout + β L_energy + γ L_reg
4. evaluate.py — Evaluation metrics:
   - Rollout MSE over long horizon
   - [0,1] bound violation rate
   - Energy monotonicity: E(u_k) <= E(u_0) ?
   - Semigroup defect: ||Phi(u_0, t+s) - Phi(Phi(u_0,t), s)||
5. run_experiments.py — Main script:
   - Generate training/validation PDE trajectories
   - Train both models
   - Evaluate both
6. plot_results.py — Visualization

Requirements:
- PyTorch, torchdiffeq (odeint with dopri5 or rk4)
- GPU support if available
- Use Python venv at /home/shuixinf/pyenvs/research/ (activate with source /home/shuixinf/pyenvs/research/bin/activate)
- All scripts should be runnable as: python run_experiments.py
- Print progress during training
- Save checkpoints and results
- Keep models small for fast iteration (< 100K params each)
- Use 1D domain, 64 grid points, dt for ODE solver at 0.01

After writing all files, do a smoke test:
  cd /home/shuixinf/projects/DC-PINNs_LaTeX_arxiv_2604.13723/experiments
  source /home/shuixinf/pyenvs/research/bin/activate
  python -c "from pde_solver import *; from models import *; print('Import OK')"
