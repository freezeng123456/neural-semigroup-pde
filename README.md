# Structure-Preserving Neural Semigroups for PDEs

Research code and LaTeX materials for learning time-evolution operators of autonomous partial differential equations at the semigroup level.

## Project goal

The project studies a learned family of evolution operators

\[
\Phi_\theta(u,\tau) \approx \mathcal S_\tau(u),
\]

where \(\mathcal S_\tau\) is the solution semigroup of a time-dependent PDE. The central idea is to learn a continuous-time latent dynamical system instead of an isolated one-step predictor:

\[
\dot z=-K_\theta(z)\nabla\Psi_\theta(z), \qquad u=g(z).
\]

Here, a bounded decoder \(g\) is intended to preserve an invariant state region, while a positive-semidefinite mobility \(K_\theta\) and latent potential \(\Psi_\theta\) provide a dissipative gradient-flow structure. Because the latent dynamics are autonomous, their exact continuous-time flow has a natural composition law, which makes the construction suitable for repeated time stepping.

The research objective is therefore broader than short-horizon prediction: learn PDE evolution families that remain useful under repeated composition while retaining interpretable structural properties such as boundedness, dissipation, and time-composition consistency.

## Current status

This is a research prototype and an active mathematical draft. The repository contains the reviewed/fixed snapshot supplied with the project, including experiments, results, model checkpoints, figures, and paper/presentation sources.

The current review identifies several items that still need to be resolved before the claims should be treated as publication-ready:

- exact semigroup composition belongs to the continuous latent ODE; an RK4 rollout has only approximate composition;
- monotonicity is established for the learned latent energy, not automatically for the physical PDE energy;
- the approximation result is currently stated more broadly than the implemented model class justifies;
- the experiments use \(\beta_V=0\), so the stated global-flow/coercivity argument does not formally cover all reported configurations;
- the corrected Fisher--KPP evaluation aligns the model horizon with the reference trajectory, but the full benchmark suite still needs to be rerun and audited;
- the Allen--Cahn and Burgers tables in the draft contain placeholder entries, and viscous Burgers is treated as an empirical out-of-class benchmark rather than a pure gradient-flow example.

See [`REVIEW_AND_FIX_REPORT.md`](REVIEW_AND_FIX_REPORT.md) and [`review_mathematical_rigor.md`](review_mathematical_rigor.md) for the detailed audit.

## Repository layout

- `experiments/` — PDE solvers, latent and baseline models, training/evaluation utilities, benchmark runners, results, and checkpoint snapshots.
- `semigroup_bound_dissipative_one_step_network_note.tex` — main mathematical note.
- `presentation.tex` — Beamer presentation source.
- `architecture.excalidraw`, `arch_diagram.html`, and image files — architecture diagrams and visual materials.
- `REVIEW_AND_FIX_REPORT.md` — focused review of the implementation and current theoretical claims.
- `review_mathematical_rigor.md` — broader mathematical-rigor review of the draft.
- `AGENTS.md`, `latex-note-formatter.agent.md`, and `latex-writing-preferences.md` — project-local writing/workflow guidance retained from the source archive. They are documentation for contributors and tooling, not additional user requirements.

## Benchmarks

The experiment scripts cover variants of:

- Fisher--KPP;
- Allen--Cahn, including two-dimensional and architecture-ablation variants;
- viscous Burgers;
- comparisons with FNO and ResNet-style baselines;
- ablations, error-growth studies, latent-parameter sweeps, and repeated-composition evaluations.

The scripts are designed around PyTorch and NumPy, with plotting and numerical utilities used by selected runners. Exact command-line arguments and defaults should be checked in each `experiments/run_*.py` file before launching a long run.

## Reproducibility notes

The supplied snapshot includes JSON result summaries and model checkpoints. Generated Python caches and LaTeX build products are excluded from future commits by `.gitignore`. Before interpreting a result, record the benchmark script, model variant, PDE parameters, time step, reference time spacing, random seed, and checkpoint path.

For a clean maintenance workflow:

1. keep source code, mathematical sources, figures, result summaries, and intentionally selected checkpoints under version control;
2. rerun evaluation after changing time alignment or rollout code;
3. report continuous-flow guarantees separately from numerical-integrator behavior;
4. distinguish architectural invariants from empirical accuracy and from PDE-level transfer theorems.

No license is asserted in this snapshot because the source archive did not provide one. Add an explicit license before distributing the repository publicly.
