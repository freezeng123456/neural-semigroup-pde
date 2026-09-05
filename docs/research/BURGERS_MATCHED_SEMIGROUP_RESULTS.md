# Matched Burgers semigroup screen: completed exploratory results

Date: 2026-09-05

The frozen three-seed matrix of `BURGERS_MATCHED_SEMIGROUP_SCREEN.md` is
complete.  This is the first transport--diffusion test after Fisher--KPP, and
the first time the autonomous-versus-query-time comparison has been run on a
PDE that is not a pure reaction--diffusion gradient flow.  All values are
`exploratory=true` and `do_not_use_for_formal=true`; they do not touch the
locked Fisher--KPP decision.

## Outcome

The primary endpoint did **not** reach the project's 10% material threshold,
while every structural endpoint favored the autonomous model.

| Readout | Observed | Reference |
|---|---:|---:|
| Geometric mean `MSE(A)/MSE(B)` over 12 cells | `0.9933176` | material threshold `0.90` |
| Seeds with per-seed geometric mean below one | `3/3` | reported |
| Cells favoring A | `7/12` | reported |
| Cells with lower A equal-work defect | `12/12` | reported |
| Geometric-mean equal-work defect ratio `A/B` | `2.908935e-7` | reported |

The registered conclusion recorded by the aggregator is
`composition_advantage_without_material_accuracy_advantage`.

The 10% threshold is not introduced here.  It is the convention already fixed
in `AUTO_RESEARCH_PROTOCOL.md` and Section 4 of
`SEMIGROUP_ATTRIBUTION_AND_TRANSFER_PROTOCOL.md`, both of which predate this
run.

## Accuracy by seed and cell

Ratios are autonomous divided by query-time, so values below one favor the
autonomous model.

| Seed | `tau=0.04, T=0.4` | `tau=0.04, T=0.8` | `tau=0.08, T=0.4` | `tau=0.08, T=0.8` | Seed GM |
|---:|---:|---:|---:|---:|---:|
| `31415` | `0.9559108` | `1.0806258` | `0.9077583` | `1.0384295` | `0.9933676` |
| `271828` | `1.0363610` | `0.9833553` | `0.9891908` | `0.9401500` | `0.9866763` |
| `161803` | `1.0098382` | `1.0424981` | `0.9500555` | `0.9996377` | `0.9999532` |

The direction is consistent at the seed level and inconsistent at the cell
level.  The magnitude, `0.67%`, is far inside the range that Fisher--KPP
already showed to be seed dependent.

## Structural endpoints

| Endpoint | Autonomous | Query-time |
|---|---:|---:|
| Equal-work RMS composition defect, GM over 12 cells | `4.717557e-8` | `1.621747e-1` |
| Spatial-mean drift, max over all cells | `7.153722e-8` | `7.313793e-8` |
| Quadratic-energy change, least negative cell mean | `-3.183289e-3` | `-3.251468e-3` |
| Positive quadratic-energy increment fraction, max | `0.04` | `0.0` |

Both models are built from a periodic discrete divergence plus the same fixed
viscous anchor, so both preserve the spatial mean to float32 roundoff.  Mean
preservation is therefore a successful implementation check for the split
generator, not a discriminator between the two temporal rules.  Both models
decreased the mean quadratic energy in every cell, and the autonomous model
was slightly *worse* on the pointwise energy endpoint: at horizon `0.8` up to
`4%` of its validation samples had a positive quadratic-energy increment,
against none for the query-time model.

## Post-hoc diagnostic: unseen-lag prediction consistency

One generator applied with matched work per unit time should reach the same
state whichever intermediate lag is requested.  The largest relative spread of
rollout MSE across the two unseen lags at a fixed horizon was
`2.474978e-7` for the autonomous model and `6.292526e-2` for the query-time
model.

This separates the two models by roughly five orders of magnitude on a
*prediction-level* readout rather than on a composition-defect readout, which
is a stronger statement of the same mechanism than the defect table alone.  It
is descriptive: it was computed after the cells completed, is labeled
`post_hoc_diagnostic` in `aggregate.json`, and takes no part in the decision
rule.

## Caveats

- **The query-time defect magnitude is inflated by out-of-range
  conditioning.** In the frozen equal-work definition, the direct path feeds
  the full horizon into the control channel, which the model normalizes by
  `max(test_taus)=0.08`.  A horizon of `0.4` or `0.8` therefore presents a
  control input of `5` or `10`, while training only ever presented
  `0.3125`--`1.25`.  The sign and existence of the query-time defect are
  meaningful, but its absolute size mixes genuine non-semigroup structure with
  extrapolation of the control channel, and it must not be compared across
  PDEs by magnitude.  The Fisher lane ran an explicit fixed-query-time control
  for exactly this reason; the Burgers lane has no such control yet.
- **The 100-epoch budget was not saturated.** All six trainings selected
  epoch 100, the last epoch, so validation MSE was still improving when the
  frozen budget ran out.  The accuracy comparison is therefore a fixed-budget
  comparison, not a converged one.
- **CPU execution.** No CUDA device was available, so the protocol's smoke
  could not confirm CUDA execution.  Every other smoke check — exact
  parameter matching at 1,313 parameters each, finite forward and backward
  values, mean preservation, checkpoint creation, parseable summary output —
  was performed on CPU.  Determinism relies on
  `torch.use_deterministic_algorithms(True)` with single-threaded BLAS,
  recorded in the run card.
- This is one one-dimensional periodic problem at `N=64` and `nu=0.01` with
  50 validation trajectories.  It is not a benchmark claim.

## Consequence for the research line

Section 4 of `SEMIGROUP_ATTRIBUTION_AND_TRANSFER_PROTOCOL.md` requires the
autonomous model to meet the accuracy and composition criteria on Allen--Cahn
*and* on one transfer PDE before the broad semigroup route is taken.
Fisher--KPP already failed that criterion in the locked 18-cell decision.
Burgers now fails the same accuracy criterion on a structurally different
generator — conservative flux plus fixed dissipation rather than a diagonal
dissipative gradient flow — while reproducing the Fisher signature exactly:
a unanimous, very large composition and lag-consistency advantage next to a
sub-percent accuracy difference.

Two independent PDEs with different generator structure now support the same
narrow claim and reject the same broad one.  The defensible statement is about
time-homogeneity as a *structural* property — composition consistency and
unseen-lag prediction consistency — and not about predictive accuracy.

The next justified steps are the fixed-query-time direct-path control named in
the caveats, which is cheap and checkpoint-only, and the mass-conserving
no-flux case (`Cahn--Hilliard`, generator $-D^\top M_\theta(u)D\mu_\theta(u)$)
recommended in `BOUNDARY_FAMILY_SEMIGROUP_WAVE2_RESULTS.md`, which is the only
remaining structure where autonomy is coupled to a physically meaningful
invariant rather than to an affine or periodic constraint.

Machine-readable values, per-seed summaries, selected checkpoints, input
hashes, and the launcher are stored under
`results/20260905-burgers-matched-semigroup-screen-e479174-3seed-cpu-r1/`.
