# Semigroup evaluator engineering audit and cleanup record

## Scope and frozen boundary

This audit covers the historical Fisher experiment/evaluation path and the new
exploratory mechanism experiments. It does not rewrite the formal training
implementation. In particular, `experiments/models.py`,
`experiments/evaluate.py`, `experiments/run_fisher_fair.py`,
`experiments/pde_solver.py`, `experiments/training.py`, and
`experiments/seed_utils.py` remain unchanged because their exact SHA-256 values
are embedded in the frozen Fisher checkpoints.

No historical result root, failed root, selected checkpoint, formal cache, or
source archive is deleted or overwritten. “Cleanup” here means removal of
duplicated source logic and replacement by a tested shared interface, not
destruction of experiment evidence.

## Historical findings

The old scripts grew by vertical accretion:

- `evaluate_fisher_mechanism.py` combined CLI parsing, output-path policy,
  hashing, JSON/CSV atomic writes, environment provenance, checkpoint schema
  reconstruction, query-time intervention, partition generation, numerical
  integration, metrics, multi-checkpoint aggregation, and receipts in one
  module of roughly 1,800 lines.
- Hashing, strict JSON, atomic Torch saves, source manifests, and input
  immutability checks were separately implemented in multiple Fisher and
  Allen--Cahn runners. The implementations were similar but not identical;
  some rejected non-finite JSON while others converted it to `null`.
- The historical equal-work helper temporarily mutated `model.ode_steps` and
  restored it in `finally`. It was correct under its tests, but the mutable seam
  made concurrent evaluation and extension to other integrators unnecessarily
  fragile.
- The existing `evaluate_fisher_mechanism_diagnostics.py` wrapper is not dead
  code. It is a compatibility adapter for direct `runpy` execution and the
  pinned SCNet runtime, so it is retained.
- The legacy Burgers runner is not a reusable formal A/B transfer experiment:
  its losses, model structures, and provenance do not match the Fisher
  intervention. The current dissipative latent generator also lacks the
  conservative/skew transport structure Burgers requires.
- There is no scientifically adequate Cahn--Hilliard solver/model pair in the
  repository. A diagonal mobility model cannot be relabeled as a
  mass-conserving fourth-order gradient flow.

## Replacement design

The new lane uses four deep modules with narrow interfaces:

1. `experiment_artifacts.py` owns the exploratory artifact contract: reserve a
   brand-new root, reject overlap with protected inputs, hash immutable inputs,
   write JSON/CSV/Torch artifacts atomically, and record runtime provenance.
2. `latent_integrators.py` owns Euler, explicit midpoint/RK2, and classical RK4
   integration plus executable RHS accounting. It evolves A or B without
   changing `model.ode_steps` or any parameter/buffer.
3. `fisher_frozen_inputs.py` is the only adapter that understands historical
   Fisher checkpoint/cache schemas. It reconstructs A/B, checks the source
   commit and every frozen training-source hash, validates cache identity and
   time grids, and exposes reference endpoints.
4. `evaluate_fisher_semigroup_grid.py` owns experiment orchestration only. One
   process evaluates one checkpoint/integrator cell, shares composed rollouts
   across horizons, reuses the identical RK4 composed trajectory between the
   production and 120-RHS semantics, and asserts equal direct/composed work.

The analytic advection--diffusion calibration calls the same explicit
integrator kernel, so an RHS-accounting bug cannot be hidden behind a separate
calibration implementation.

## Deleted or consolidated redundancy

`evaluate_fisher_mechanism.py` no longer carries private implementations of:

- the protected-root and input-overlap guard;
- file SHA-256;
- Git commit discovery;
- device/runtime provenance;
- tensor/NumPy-to-strict-JSON conversion;
- atomic JSON and CSV replacement;
- input before/after immutability verification.

It imports these through compatibility aliases, preserving its tested public
surface while deleting the duplicated implementations. The frozen formal
source files were deliberately not migrated.

## New execution path

- `prepare_fisher_phase_cache.py` is the only phase-lane cache writer. Its CLI
  has one configurable value, the new output root; seed, grid, PDE parameters,
  sample count, reference step, and maximum horizon are frozen in code. It uses
  one vectorized spectral time loop, atomically saves the cache, reloads it,
  validates every trajectory, and records the resulting digest.
- `evaluate_fisher_semigroup_grid.py` has no cache-creation or training code.
  It requires explicit checkpoint, cache, and source-archive hashes, and
  re-hashes all three inputs after inference.
- `calibrate_linear_advection_diffusion.py` is an analytic numerical control,
  not a trained cross-PDE result.

## Verification gates

The new tests cover:

- exact 120-RHS conversion to 120/60/30 Euler/RK2/RK4 substeps;
- numerical equivalence of the new RK4 seam to both historical A and B forward
  paths, with parameter/buffer and `ode_steps` immutability;
- per-cell equal-RHS assertions and integer composition depths;
- unique-root rejection and strict JSON output;
- checkpoint reconstruction against the frozen training source hashes;
- formal cache identity and reference-grid validation;
- the fixed phase-cache protocol;
- exact Fourier semigroup composition and observed first-, second-, and
  fourth-order convergence.

Local syntax and lint checks do not require PyTorch. The repository tests and
all model/cache smoke checks must run in the pinned SCNet PyTorch environment
before any full GPU evaluator is submitted.

## Deliberately deferred work

- Migrating formal runners to the shared modules would change frozen source
  hashes and is therefore rejected for this branch.
- Generalizing the new evaluator to arbitrary checkpoints is rejected: the
  current interface is intentionally deep and specific to the preregistered
  Fisher A/B intervention.
- Burgers and Cahn--Hilliard require new mathematical model structures and
  paired protocols; they are not engineering-only extensions of this lane.
