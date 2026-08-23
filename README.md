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

The major-revision branch now resolves the review's central formulation and implementation issues:

- exact continuous flows, numerical RK4 maps, and their numerical composition defects are named separately;
- learned-energy and optional physical-energy diagnostics use distinct result fields;
- the transfer theorem now uses explicit projection/reconstruction maps and continuous Grönwall comparison, with numerical integration errors added separately;
- the former universality statement is replaced by a proved componentwise error bound for architecture-compatible generators;
- strict reference-time alignment is enforced in both evaluation and training-time validation;
- a checkpoint-compatible `beta_V_floor` option supports genuinely coercive new configurations;
- unified seeding, an architecture-only loss mode, and focused regression tests have been added;
- missing Allen--Cahn/Burgers values are labeled “Not reported,” and Burgers is treated as an out-of-class transport stress test.

The repository is not yet publication-ready: all comparative benchmark tables still require aligned, multi-seed reruns. Archived numerical values are retained for provenance and are not presented as corrected results.

See [`REVIEW_AND_FIX_REPORT.md`](REVIEW_AND_FIX_REPORT.md) and [`review_mathematical_rigor.md`](review_mathematical_rigor.md) for the detailed audit.

The concrete mathematical, code, validation, and rerun plan is documented in [`MAJOR_REVISION_SOLUTION.md`](MAJOR_REVISION_SOLUTION.md).

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

## Verification and revised Fisher--KPP run

Create an environment with PyTorch, NumPy, and pytest, then run:

```bash
python3 -m pytest experiments/tests -q
python3 -m compileall -q experiments
```

The revised Fisher--KPP runner exposes the controls needed by the review:

```bash
cd experiments
python3 run_experiments.py \
  --data-seed 42 \
  --seed 42 \
  --deterministic \
  --architecture-only \
  --beta-v-floor 0.1 \
  --no-resume \
  --checkpoint-dir checkpoints/fisher_kpp_revised_seed42 \
  --results-dir results/fisher_kpp_revised_seed42
```

Use distinct output directories for every seed and configuration. A positive `--beta-v-floor` is required when a run is claimed to fall under the coercive global-flow criterion; it must not be conflated with archived `beta_V=0` checkpoints.

No license is asserted in this snapshot because the source archive did not provide one. Add an explicit license before distributing the repository publicly.
