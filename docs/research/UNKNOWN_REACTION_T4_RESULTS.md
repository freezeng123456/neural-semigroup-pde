# Shared physics, unknown reaction: completed T4 identification screens

2026-09-07. Both matrices are complete: 48 full training cells, plus two
four-cell smoke runs. All evidence is exploratory. No formal Fisher result
is changed, and the post-screen ablation is not an independent confirmation.

## Outcome

Providing the same known cubic dissipation to both learned models reduces
rollout MSE by about 100-fold and removes observed state-bound violations.
Autonomy passes the composition-refinement rule in both phases but fails the
preregistered material-accuracy rule in both. The main useful result is the
importance of the shared dissipative prior in this small identification
problem, not a general prediction advantage of neural semigroups.

## Design and execution

The new fixed-grid periodic PDE is
`u_t = 0.02 D2 u + u - u^3 + 0.2 sin(3u)`, on [0,2*pi), N=64.
All models receive the same exact discrete heat operator. Phase 1 learns the
entire reaction. Phase 2 supplies the same known `-u^3` term to both models
and learns the remaining source from state pairs only. The reaction/source
formula is unavailable to training except for the explicitly supplied cubic.

- A: autonomous reaction network with an inactive duration input.
- B: requested-duration-conditioned reaction network.
- Same 65 parameters and initial weights within every paired cell.
- At one evaluation, B is exactly a physics-split direct increment map;
  there is no separately trained, numerically identical C baseline.
- Three seeds x training pairs {16,128} x internal substeps {1,4} x {A,B}.
- 120 full-batch Adam updates, lr=0.01, selection by one-step validation only.
- Independent validation/test initial states; training subsets nested.
- Primary: GM MSE over unseen lags {0.075,0.15}, horizons {1.2,2.4}.
- Oracle RK4 is FP64 at dt=0.002. Halving dt on eight test paths gives max
  relative L2 `4.3089560281915936e-13`, below the frozen `1e-5` threshold.
- Single Tesla T4, PyTorch 2.0.1+cu118; each matrix occupied one process.
  Source-to-done durations were 72 and 70 seconds, including preparation and
  evaluation. Summed per-cell training times were 38.67 and 39.29 seconds.

Source commits:

- Phase 1: `40616f3d59eb64f10b61a735a5966586c39146dc`.
- Phase 2: `b03c41e5f992beb7329feeb1ef852daf15a0653f`.

Shared full cache SHA256:
`9b761475a354580a7f597a88d8715cd5b57037d5c669a5c8fc3e8faf28bda8d5`.

## Main endpoints

| Training pairs | Substeps | Phase 1 A/B | Phase 2 A/B | Phase 2 A MSE | Phase 2 B MSE |
|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 1.02672 | 0.82494 | 0.0117191 | 0.0142060 |
| 16 | 4 | 0.97381 | 0.95312 | 0.0083813 | 0.0087935 |
| 128 | 1 | 0.97452 | 1.00513 | 0.0114071 | 0.0113489 |
| 128 | 4 | 1.05650 | 1.01106 | 0.0084836 | 0.0083908 |
| Pooled | — | **1.00727** | **0.94546** | **0.00987395** | **0.01044353** |

The material A/B rule requires pooled <=0.90, at least two seed-level ratios
<=0.90, no stratum >1.05, and the stated physical-diagnostic noninferiority.
Phase 2 seed ratios are 1.01618, 0.86649, 0.95983 for seeds 31415, 271828,
161803 respectively. Only one seed clears 0.90. The low-data/one-evaluation
stratum is a descriptive signal, not grounds to replace the pooled rule.

Absolute paired ablation:

| Readout | A | B |
|---|---:|---:|
| Phase 1 pooled MSE | 1.0932546 | 1.0853612 |
| Phase 2 pooled MSE | 0.00987395 | 0.01044353 |
| Phase 2 / Phase 1 MSE | **0.00903170** | **0.00962217** |
| Phase 1 mean bound-violation fraction | 0.3645908 | 0.3659697 |
| Phase 2 mean bound-violation fraction | 0.0 | 0.0 |

Every seed improves materially under the anchor, for both A and B. Thus the
separately frozen shared-dissipation rule passes. The bound fraction counts
evaluated space-time coordinates across all tested lags, including 0.30; it
is not the fraction of failed trajectories. No sample was dropped.

## What failed in Phase 1, and what remains imperfect

All 24 selected scalar reaction curves point outward at every sampled point
with |u|>=1.05 on [-1.2,1.2] at conditioning duration 0.15, whereas the true
reaction points inward there. Phase 1 fits one-step validation at about
2.5e-4 to 3.0e-4 yet drifts away on long rollouts. Both learned models are
worse than pure heat (primary MSE 0.317551). This is failed generator
identification, not merely a close accuracy race between good predictors.

Phase 2 supplies genuinely additional physical information. Its improvement
does not establish recovery of a completely unknown reaction under Phase 1's
information budget. It does establish the usefulness of this shared prior
in the tested representation. Both temporal models benefit comparably.

The no-learning known-cubic-only baselines have MSE 0.377416 and 0.376401 at
budgets 1 and 4. The learned source therefore does useful work after anchoring.
The hidden-reaction oracle splits achieve 2.03313e-4 and 1.26184e-5. They
have privileged information and are diagnostic floors, not fair learned
competitors. At budget 4, learned errors remain roughly 660 times the oracle
MSE. Refining the frozen budget-4 models to 16 steps only reduces their MSE
by about 3.5% (A) and 3.3% (B). More integration work does not close the gap.

Observed zero violations of |u|<=1.2 do not constitute an invariance proof.
Physical energy is not monotone on every evaluated increment: the smallest
cell-level monotone fractions in Phase 2 are 0.82868 (A) and 0.78446 (B).
This architecture does not certify the target physical energy law.

## Composition result

The frozen budget-4 checkpoints were evaluated on two unequal two-part
compositions and one four-part composition, with 1,4,16,64 substeps per call.
The paths have deliberately different step sizes; their evaluation counts
are recorded. This is a refinement diagnostic, not the equal-work primary
accuracy comparison.

| Internal steps | Phase 2 A relative RMS defect | Phase 2 B relative RMS defect |
|---:|---:|---:|
| 1 | 0.00864597 | 0.01011567 |
| 4 | 0.00235046 | 0.00457753 |
| 16 | 0.000600122 | 0.00344131 |
| 64 | 0.000142562 | 0.00322912 |

A's 1-to-64 reduction is monotone and ranges from 58.3 to 62.7 in every
seed/size/composition cell, above the frozen gain-20 threshold. Phase 1 also
passes (54.3 to 61.5). This agrees with first-order reaction Euler error;
the heat-half-step sandwich is NOT a second-order Strang scheme because its
reaction substep is first order. B's residual approaches a nonzero floor in
the observed sweep. No finite sweep proves its exact limiting value.

## Validation and reproducibility

Six focused tests pass on T4; each code baseline also passed a four-cell GPU
smoke. Both full launchers exited 0 with 24 done cells and no failed marker.
Independent postprocessing verified 48 selected-checkpoint hashes, shared
cache integrity, paired initialization within and across phases, all 5,760
epoch records, validation-only selection, and recomputed both A/B aggregates.
The full roots include all endpoint/sample MSEs, 96 full-run checkpoints
(best/final), CSVs, logs, configs, strict JSON decisions, caches, and figures.

Canonical roots are `/data/semigroup_runs/` followed by:

- `20260907-unknown-reaction-40616f3-t4b-full-r1`
- `20260907-unknown-reaction-anchor-b03c41e-t4b-full-r1`

The same names appear under `results/` in the publication branch. Smoke roots
are retained separately and never pooled with these results. Analysis scripts
are `experiments/analyze_unknown_reaction_matched.py` and
`experiments/compare_unknown_reaction_phases.py`.

## Research decision

Stop additional training and sweeps for this task. Retain the shared known
dissipation + learned unknown source as a viable exploratory formulation,
but reject a material A/B accuracy claim. Autonomy currently earns the
refinable composition property, not a broadly superior predictor claim.

The next useful experiment would isolate supervision on states visited during
long trajectories, on a new held-out dataset, with both temporal models given
the same dissipative information. A stable and accurate learned generator
must be established before testing a semigroup-specific benefit. Keep the
oracle and incomplete-physics baselines, report physical-energy defects, and
do not scale this one-dimensional result to cross-PDE or publication claims.
