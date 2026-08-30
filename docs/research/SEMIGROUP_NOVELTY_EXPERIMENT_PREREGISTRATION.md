# Semigroup novelty mechanism experiments: exploratory preregistration

**Frozen on:** 2026-08-30, before inspecting any result from the experiments
defined below.

## Status and claim boundary

Every experiment in this document is **exploratory mechanism evidence**. None
changes, extends, or re-decides the Allen--Cahn or Fisher--KPP formal protocols.
The formal checkpoints and locked Fisher cache remain read-only, no checkpoint
is selected from these experiments, and every execution uses a new canonical
root.

The target claim is deliberately narrower than algorithmic priority:

> For the tested autonomous reaction--diffusion systems, integrating one
> state-only generator can produce substantially more composition-consistent
> learned maps than conditioning the generator on the requested time. The
> effect must survive numerical-integrator and work-budget controls, while
> prediction accuracy is reported as a separate endpoint.

This document does not preregister claims that this is the first neural
semigroup, that a small composition defect guarantees a small rollout error,
or that the architecture applies unchanged to non-autonomous, transport, or
mass-conserving PDEs.

## Frozen provenance and non-selection rules

- Fisher training source commit:
  `637345584dc2db8ddccf9116a995615c3c036104`.
- Fisher training source archive SHA-256:
  `a370efa4af9bbb11fbcd72ef422410f651f8f2ad28eee561c45e766c88fcadf5`.
- Formal Fisher locked-cache SHA-256, used only by the integrator control:
  `29d0e4b36e9d758f87264ae1d555d865c9e037776903e1cab448a0a8782b490e`.
- Model checkpoints are the already validation-selected formal A/B checkpoints
  for seeds `31415`, `271828`, and `161803`. Their SHA-256 values must be
  recomputed at launch and again after evaluation.
- No test or exploratory result may select, replace, tune, average, or modify a
  checkpoint.
- The phase-diagram cache is new and explicitly exploratory. It is never copied
  over the formal cache and is not used to recompute the formal Fisher decision.
- Failed and superseded roots are retained and never reused.

## Experiment I: cross-integrator and equal-work control

### Matrix

- models: formal Fisher A (`latent`) and B (`latent_query_time`);
- seeds: `31415`, `271828`, `161803`;
- unseen lags: `0.075`, `0.15`;
- horizons: `1.2`, `2.4`, `4.8`;
- integrators: forward Euler, explicit midpoint/RK2, and classical RK4;
- samples: all 500 trajectories in the immutable formal locked cache;
- training or fine-tuning: none.

Each seed/model/integrator cell is an independent checkpoint-only evaluator.
It reports both of the following numerical semantics:

1. **Production fixed-substep:** 30 substeps per model call, preserving the
   historical inference convention. Euler, RK2, and RK4 therefore have 1, 2,
   and 4 right-hand-side evaluations per substep.
2. **Equal-RHS-work:** 120 right-hand-side evaluations per base-lag call.
   Euler uses 120 substeps, RK2 uses 60, and RK4 uses 30. A direct path over a
   complete horizon receives exactly the same total number of right-hand-side
   evaluations as the corresponding repeated-lag composition. The evaluator
   must record and assert equality, not infer it from configuration labels.

### Endpoints

- rollout MSE and relative L2;
- production fixed-substep composition defect;
- equal-RHS-work composition defect;
- direct and composed right-hand-side evaluation counts;
- finite-value and failure counts;
- checkpoint, cache, training-source archive, evaluator source, result, and
  receipt hashes.

### Advance/falsification rule

Cross-integrator robustness passes if A's aggregated equal-work defect is
lower than B's for at least two of the three integrators, the same direction
holds for at least two of three seeds within each passing integrator, and every
reported equal-work cell has exactly matched direct/composed RHS counts.

Rollout MSE is not part of this structural pass rule. Its direction is reported
separately. If the defect advantage occurs only under RK4, the proposed
structural interpretation is rejected as an integrator-specific artifact.

## Experiment II: Fisher lag--horizon phase diagram

### New exploratory cache

- PDE: periodic 1D Fisher--KPP;
- `N=64`, `L=10`, `nu=0.1`, reaction rate `r=1`;
- reference solver step: `0.005`;
- exploratory test seed: `314164`;
- samples: `128`;
- maximum cached horizon: `4.8`;
- cache creation: one explicit prepare-only job with atomic creation and a
  post-write reload/validation pass.

The cache configuration, tensor shapes, timestamps, finite values, admissible
state range, byte size, and SHA-256 are recorded before any evaluator starts.

### Matrix

- frozen formal A/B checkpoints for the three formal seeds;
- lags: `0.025`, `0.05`, `0.075`, `0.10`, `0.15`, `0.20`, `0.30`;
- horizons: `0.6`, `1.2`, `2.4`, `4.8`;
- RK4 with 30 substeps per base-lag call, plus an equal-work direct path;
- no `0.0125` lag and no `9.6` horizon in this first screen.

Every horizon/lag pair must have an integer composition depth and align with
the cache reference step. A later `9.6` extension requires a separately frozen
cache and is not authorized by this document.

### Endpoints

- per-seed and aggregate rollout MSE and relative L2;
- direct-versus-composed relative-L2 and MSE defects;
- composition depth `H/tau` and direct/composed RHS counts;
- median, 90th percentile, 95th percentile, worst-sample value, and finite
  failure count;
- a complete machine-readable grid, not only a heatmap.

### Advance/falsification rule

The composition-depth hypothesis passes only if A has lower equal-work defect
in at least 80% of valid grid cells for at least two of three seeds and B's
defect has a positive rank association with composition depth for at least two
of three seeds. The association statistic and all cells are reported even if
the rule fails. MSE is not allowed to substitute for the defect criterion.

## Experiment III: analytic advection--diffusion calibration

Use the periodic linear equation

\[
u_t + c u_x = \nu u_{xx}
\]

only as a numerical calibration, not as a trained cross-PDE result. The exact
Fourier evolution is

\[
\widehat{u}_k(t)=\exp\!\left((-i c k-\nu k^2)t\right)\widehat{u}_k(0).
\]

The calibration must compare the exact direct and composed flows and measure
Euler, midpoint/RK2, and RK4 convergence under explicitly reported step and RHS
budgets. It passes if the exact-flow composition defect is at floating-point
roundoff, errors decrease under refinement for all three methods, and the
observed order is consistent with first, second, and fourth order within the
predefined test tolerances. This test validates the accounting and metric; it
does not show that the learned Fisher A/B models transfer to transport.

## Deferred experiments

- The legacy Burgers runner is not accepted as an A/B semigroup experiment:
  its losses and provenance are not matched, and the current purely
  dissipative generator lacks a transport/skew component.
- Cahn--Hilliard is deferred until a conservative positive operator, mass
  preservation, chemical-potential structure, and mass/energy endpoints are
  implemented.
- No large data-scaling, parameter-OOD, Navier--Stokes, or
  Kuramoto--Sivashinsky wave is launched from partial results of this screen.

## Completion gate

An experiment is complete only after normal process exit, parseable finite
outputs, all requested cells, matching input hashes before and after the run,
an immutable receipt/manifest, recorded RHS budgets, and successful independent
aggregation. A submitted job, a final log line, or a partial heatmap is not
completion.
