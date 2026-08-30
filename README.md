# Structure-Preserving Neural Semigroups for PDEs

Research code and LaTeX materials for learning boundary-admissible
time-evolution operators of autonomous partial differential equations.

## Project goal

The project studies a learned family of evolution operators

\[
S_t^\theta:X_B\to X_B,
\qquad
X_B=\{u:B_hu=g\},
\]

where `X_B` is a discrete state space satisfying a selected spatial boundary
condition. The main construction combines two distinct structures:

1. a parameter-free boundary map and tangent vector field keep the learned
   flow inside `X_B`;
2. a duration-independent autonomous generator supplies time homogeneity and
   the continuous-flow composition law.

The semigroup law does not create the spatial boundary condition. The
boundary operator defines the domain on which the semigroup acts. Boundary
residual, numerical composition defect, and PDE prediction error are therefore
measured separately.

One generator family in the repository is a continuous-time latent
gradient-flow system:

\[
\dot z=-K_\theta(z)\nabla\Psi_\theta(z), \qquad u=g(z).
\]

Here, a bounded decoder \(g\) is intended to preserve an invariant state region, while a positive-semidefinite mobility \(K_\theta\) and latent potential \(\Psi_\theta\) provide a dissipative gradient-flow structure. Because the latent dynamics are autonomous, their exact continuous-time flow has a natural composition law, which makes the construction suitable for repeated time stepping.

The research objective is broader than short-horizon prediction: learn PDE
evolution families that remain useful under repeated composition while
retaining explicit spatial admissibility and, when the generator supports
them, interpretable bounds, invariants, or dissipation.

## Theorem hierarchy

The active mathematical draft now proves the fixed-grid architecture layer:

- the Dirichlet, homogeneous-Neumann, and Robin endpoint reconstructions are
  affine retractions onto their stated discrete boundary spaces;
- the corresponding tangent maps place every hard-mode derivative in the
  homogeneous boundary space;
- for fixed finite weights, the autonomous Tanh generator is globally
  Lipschitz and therefore defines a unique global forward semiflow;
- that exact flow preserves the selected discrete boundary relation and
  satisfies the composition law;
- every explicit Runge--Kutta stage preserves the same affine relation in exact
  arithmetic;
- under standard smoothness and stability hypotheses, nonuniform RK4
  direct-versus-composed defects are fourth order in the largest substep.

These results do not by themselves prove agreement with a continuous PDE.
The PDE layer additionally requires boundary consistency under mesh
refinement, method-of-lines convergence, generator matching on a common
trajectory set, and a stability or one-sided Lipschitz estimate. The derived
rollout bound keeps those terms separate. See
[the boundary-admissible theory](boundary_admissible_semigroup_theory.tex)
and the required
[PDE theorem card](docs/research/PDE_THEOREM_CARD.md).

## Current status

This is a research prototype and an active mathematical draft. The repository contains the reviewed/fixed snapshot supplied with the project, including experiments, results, model checkpoints, figures, and paper/presentation sources.

The major-revision branch now resolves the review's central formulation and implementation issues:

- exact continuous flows, numerical RK4 maps, and their numerical composition defects are named separately;
- learned-energy and optional physical-energy diagnostics use distinct result fields;
- the transfer theorem now uses explicit projection/reconstruction maps and continuous Grönwall comparison, with numerical integration errors added separately;
- the former universality statement is replaced by a proved componentwise error bound for architecture-compatible generators;
- the boundary-family code now has a matching finite-dimensional theorem for
  affine boundary invariance, global autonomous flow, RK stage preservation,
  and conditional RK4 composition convergence;
- a one-sided stability--generator-consistency estimate identifies the extra
  hypothesis needed to turn generator error into a long-horizon PDE bound;
- strict reference-time alignment is enforced in both evaluation and training-time validation;
- a checkpoint-compatible `beta_V_floor` option supports genuinely coercive new configurations;
- unified seeding, an architecture-only loss mode, and focused regression tests have been added;
- compatible validation trajectories and formal-evaluation samples are grouped
  into inference batches, while full training validation defaults to every five
  epochs with a mandatory final-epoch check;
- missing Allen--Cahn/Burgers values are labeled “Not reported,” and Burgers is treated as an out-of-class transport stress test.

The frozen boundary-family Wave 2 exploratory matrix is now complete: 54/54
full cells cover nonhomogeneous Dirichlet, homogeneous Neumann, and Robin
conditions; hard, penalty, and unconstrained enforcement; autonomous and
query-time temporal rules; and three paired seeds. All hard boundary gates
passed, and each boundary family passed the preregistered temporal rule. See
[`docs/research/BOUNDARY_FAMILY_SEMIGROUP_WAVE2_RESULTS.md`](docs/research/BOUNDARY_FAMILY_SEMIGROUP_WAVE2_RESULTS.md).

The repository is not yet publication-ready. Wave 2 is exploratory and tests
one one-dimensional reaction--diffusion PDE. The formal Fisher--KPP lane does
not establish a stable prediction advantage, and conservative/no-flux,
multidimensional, and irregular-geometry transfer remain open. Archived
numerical values are retained for provenance and must not be mixed across
formal and exploratory evidence classes.

Before broadening to another PDE, complete the theorem card. In particular,
Cahn--Hilliard requires a conservative mobility, Burgers requires a
transport/skew component, and an explicitly time-dependent PDE requires an
evolution family \(U(t,s)\) or an augmented clock state rather than the
one-parameter autonomous-semigroup claim.

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
  --validation-interval 5 \
  --no-resume \
  --checkpoint-dir checkpoints/fisher_kpp_revised_seed42 \
  --results-dir results/fisher_kpp_revised_seed42
```

Use distinct output directories for every seed and configuration. A positive `--beta-v-floor` is required when a run is claimed to fall under the coercive global-flow criterion; it must not be conflated with archived `beta_V=0` checkpoints.

`--validation-interval 5` records skipped validation epochs as `NaN` together
with a `validation_performed` mask. The final epoch is always validated. The
batched evaluator preserves per-sample statistics, strict timestamp checks,
variable rollout horizons, and numerical-semigroup-defect semantics.

No license is asserted in this snapshot because the source archive did not provide one. Add an explicit license before distributing the repository publicly.
