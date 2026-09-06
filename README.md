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

The matched Burgers screen adds a second, structurally different PDE: both
models use a periodic conservative flux plus the same fixed viscous anchor
rather than a diagonal dissipative gradient flow. Its three-seed exploratory
matrix reproduces the Fisher--KPP signature. The autonomous model has a lower
equal-work composition defect in 12/12 cells and an unseen-lag prediction
spread roughly five orders of magnitude smaller, but its geometric-mean
rollout-MSE ratio is `0.9933`, well short of the 10% material threshold. See
[`docs/research/BURGERS_MATCHED_SEMIGROUP_RESULTS.md`](docs/research/BURGERS_MATCHED_SEMIGROUP_RESULTS.md).

A checkpoint-only decomposition then separated that composition defect into a
conditioning artifact and a genuine structural term. Under matched
conditioning the query-time defect collapses to `0.94` of the autonomous
integrator floor, so the screen's headline ratio is not the structural effect
size; but an in-range cross-lag composition at equal work still leaves `1.26%`
of the state norm against exactly zero for the autonomous model. The mechanism
claim survives with a restated magnitude, and the same restatement is now owed
to the Fisher composition-defect numbers. See
[`docs/research/BURGERS_QUERY_CONDITIONING_ATTRIBUTION_RESULTS.md`](docs/research/BURGERS_QUERY_CONDITIONING_ATTRIBUTION_RESULTS.md).

A seven-variant ablation of the flux generator then showed that the screen's
architecture is the worst member of its own grid. The exact Burgers flux is
pointwise, and shrinking the flux stencil from five points to one halves
rollout error and cuts one-step error about fivefold; hidden width 32 and the
second hidden layer are pure overhead. A 33-parameter generator, `39.8` times
smaller, is about twice as accurate as the screen's 1,313-parameter one. The
fixed viscous anchor, by contrast, is essential. See
[`docs/research/BURGERS_FLUX_ABLATION_RESULTS.md`](docs/research/BURGERS_FLUX_ABLATION_RESULTS.md).

Re-running the paired contrast on that 33-parameter generator settled what the
reduction costs. The structural result survives untouched: an in-range
composition defect of `0.7456%` of the state norm against exactly zero, still
structural in all six cells, so the mechanism holds at two architectures forty
times apart. The accuracy result stays sub-material but reverses sign to
`1.0314`, which withdraws the screen's seed-level direction sub-claim and
leaves the accuracy endpoint carrying no signal either way. The reason is
visible in the selection metric: the control channel is worth `1.2%` of
one-step error at 1,313 parameters and `39%` at 33, because reading the
requested lag is a fitting shortcut whose value grows as capacity shrinks and
whose structural cost does not. See
[`docs/research/BURGERS_MINIMAL_GENERATOR_AB_RESULTS.md`](docs/research/BURGERS_MINIMAL_GENERATOR_AB_RESULTS.md).

Two further controls close the Burgers lane. Against a direct
time-conditioned map of the same 33 parameters, the in-range cross-lag defect
is `0` exactly for the autonomous flow, `0.7456%` of the state norm for the
query-conditioned flow, and `2.8013%` for the direct map, with unseen-lag
prediction spreads of `3.0e-7`, `8.0e-2` and `7.4e-1`; that ordering is
monotone in how much freedom the requested duration is given and cannot be a
compute artifact. The accompanying accuracy figure could be, and was: at
matched work, one flux evaluation per call for all three, the ratios collapse
to `1.0084` and `1.0013`, so the flow's apparent `30%` advantage was compute
and has been retracted. Nothing in the Burgers lane supports an accuracy claim
in either direction. See
[`docs/research/BURGERS_DIRECT_MAP_CONTROL_RESULTS.md`](docs/research/BURGERS_DIRECT_MAP_CONTROL_RESULTS.md)
and
[`docs/research/BURGERS_WORK_MATCHED_CONTROL_RESULTS.md`](docs/research/BURGERS_WORK_MATCHED_CONTROL_RESULTS.md).

For the adopted generator the two constants
[`docs/research/FISHER_KPP_THEOREM_CARD.md`](docs/research/FISHER_KPP_THEOREM_CARD.md)
marks `OPEN` are now certified rather than sampled, and the layered rollout
bound is a number instead of a schema. Its looseness is almost entirely one
constant: the exponential amplification costs about `20` at horizon `0.4` and
`870` at `0.8`, while generator matching costs `7` and `9`. See
[`docs/research/BURGERS_CERTIFIED_CONSTANTS.md`](docs/research/BURGERS_CERTIFIED_CONSTANTS.md).

The missing Allen--Cahn A-versus-C comparison was first run at matched
work and appeared to fire (`MSE(A')/MSE(C) = 0.756`).  That figure is
withdrawn.  The losing map double-counted diffusion.  Against a
heat-plus-reaction split that does not, the Euler flow loses by a
pooled ratio `45.1`, and a zero-parameter physics split is stronger
than the learned residual.  The same zero-parameter split, evaluated
on the locked Fisher--KPP cache, gives
`MSE(C_phys)/MSE(A) = 0.0737` in every published formal cell.  Both
1D reaction--diffusion PDEs in Section 4 are therefore classical-split
accuracy results, not semigroup results.  The Burgers work-matched
retraction and the formal Fisher A-versus-B decision are unchanged.
See
[`docs/research/ALLEN_CAHN_WORK_MATCHED_DIRECT_MAP_RESULTS.md`](docs/research/ALLEN_CAHN_WORK_MATCHED_DIRECT_MAP_RESULTS.md),
[`docs/research/ALLEN_CAHN_REACTION_ONLY_DIRECT_MAP_RESULTS.md`](docs/research/ALLEN_CAHN_REACTION_ONLY_DIRECT_MAP_RESULTS.md),
and
[`docs/research/FISHER_PHYSICS_SPLIT_RESULTS.md`](docs/research/FISHER_PHYSICS_SPLIT_RESULTS.md).
The analogous large-step conservative split on Cahn--Hilliard is not a
usable bar: it diverges at the same unseen lags.  See
[`docs/research/CAHN_HILLIARD_PHYSICS_SPLIT_RESULTS.md`](docs/research/CAHN_HILLIARD_PHYSICS_SPLIT_RESULTS.md).

The repository is not yet publication-ready. Wave 2 is exploratory and tests
one one-dimensional reaction--diffusion PDE. The formal Fisher--KPP lane and
the exploratory Burgers screen still lack a material prediction advantage
against a sibling learned map, and both reaction--diffusion PDEs lose to a
zero-parameter heat-plus-reaction split, so the Section 4 accuracy advance
criterion remains unmet as a semigroup claim. Conservative/no-flux,
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
