# Minimal Burgers generator A/B: completed exploratory results

Date: 2026-09-05

The frozen lane in `BURGERS_MINIMAL_GENERATOR_AB_PROTOCOL.md` is complete: one
paired autonomous/query-time comparison at the ablation's recommended
33-parameter generator, three seeds, everything else at the screen's frozen
values.  All values are `exploratory=true` and `do_not_use_for_formal=true`.

## Outcome

Both frozen conclusions are unchanged, so the aggregator records
`adopt_reduction: true`.  The structural result survives a forty-fold
parameter reduction; the accuracy result stays sub-material but **reverses
direction**.

| Readout | Oversized `r2_h32_l2_anchor` | Minimal `r0_h8_l1_anchor` |
|---|---:|---:|
| Parameters per model | `1313` | `33` |
| Geometric-mean `MSE(A)/MSE(B)` | `0.9933176` | `1.0314024` |
| Cells favoring A | `7/12` | `0/12` |
| Seeds meeting the `0.90` threshold | `0/3` | `0/3` |
| In-range cross-lag defect, A | `0.0` exactly | `0.0` exactly |
| In-range cross-lag defect, B | `5.751797e-3` | `3.342645e-3` |
| B in-range defect over integrator floor | `1.22e+5` | `9.01e+4` |
| B in-range defect relative to state RMS | `1.2586%` | `0.7456%` |
| In-range cells judged `structural` | `6/6` | `6/6` |

## The structural claim survives the reduction

With patch radius zero the flux network reads exactly two inputs, the local
state and the control channel, so the control channel could easily have become
ignorable.  It did not.  The query-time model still breaks in-range
composition by `0.7456%` of the state norm, `90,113` times the integrator
floor, structural in all six cells, while the autonomous model's two in-range
paths remain bitwise identical.

The prediction-level version of the same statement is equally clean: the
largest relative spread of rollout MSE across the two unseen lags at a fixed
horizon is `2.978145e-7` for the autonomous model and `7.963157e-2` for the
query-time model.

The mechanism is therefore not an artifact of the oversized architecture.  It
now holds at two architectures forty times apart in parameter count and with
structurally different flux supports, one five-point and one pointwise.

## The accuracy direction reverses, and the screen's sub-claim does not survive

The frozen rule asks only whether the `0.90` material threshold is met.  It is
not met at either architecture, so the conclusion "no material accuracy
advantage" is genuinely unchanged.  But the sign of the sub-material
difference flipped: the autonomous model was ahead in `7/12` cells and all
three seeds on the oversized generator, and is ahead in `0/12` cells and no
seed on the minimal one, losing by `3.14%`.

**The screen's "consistent direction at the seed level, 3/3" sub-claim must
therefore be withdrawn.**  It is an artifact of one architecture.  Taking both
runs together, the accuracy endpoint carries no signal in either direction:
both magnitudes are far inside the materiality band and the two architectures
disagree on sign.

## Why the direction flips: the control channel is a fitting shortcut

The clearest evidence is the selection metric, mean one-step validation MSE on
the training lags, which is the only quantity checkpoint selection ever sees.

| Architecture | A one-step | B one-step | B relative to A |
|---|---:|---:|---:|
| `r2_h32_l2_anchor` (`1313` parameters) | `3.774541e-6` | `3.727938e-6` | `0.988` |
| `r0_h8_l1_anchor` (`33` parameters) | `2.493336e-6` | `1.528587e-6` | `0.613` |

At 1,313 parameters the control channel buys essentially nothing on the metric
that selects checkpoints.  At 33 parameters it buys `39%`.

This is not a capacity difference.  Both models have exactly the same
parameter count, because the control column exists in both and model A merely
zeroes it; the difference is *input information*.  The training targets sit at
variable lags `0.025`--`0.1`, so a flux that reads the requested lag has a
legitimate shortcut for one-step fitting — and that shortcut is precisely what
destroys composition, since a lag-dependent vector field is not the generator
of one flow.

So the trade is capacity dependent.  With ample capacity the autonomous model
learns a good enough vector field that the shortcut is worthless, and the two
models tie with a slight edge to autonomy.  With 33 parameters the shortcut
becomes worth `39%` of one-step error and `3%` of rollout error, while the
composition penalty stays at the same order.  The structural cost of the
shortcut is robust; its accuracy benefit is not.

## Reproducibility across lanes

Model A of this lane is the same object as the ablation's `r0_h8_l1_anchor`
cell, trained independently by a different runner.  For all three seeds the
selected weights are **bitwise identical** and the selected epoch and one-step
validation MSE agree exactly.  The stored checkpoint digests differ only
because each runner records its own `label` string in the payload.

The reduction's accuracy gain also replicates inside the paired run: model A's
pooled rollout MSE is `0.3805` of the oversized screen's model A at horizon
`0.4` and `0.6489` at horizon `0.8`.

## Structural endpoints

Spatial-mean drift is at most `7.870e-8` for A and `6.892e-8` for B, so the
divergence form survives the reduction.  Both models decrease the mean
quadratic energy in every cell, at most `-5.2006e-3` for A and `-5.2161e-3`
for B, with no positive-mean cell.  The energy endpoint remains limited by the
reference data's own dealiasing problem recorded in
`BURGERS_FLUX_ABLATION_RESULTS.md`, so only sign-level readings are used.

## Caveats

- **Fixed budget.** All six trainings selected epoch 100, the last epoch.
  Every accuracy comparison in the Burgers lane is at equal epochs, not at
  convergence.
- The adoption verdict concerns the architecture only.  It does not restore
  the withdrawn direction sub-claim.
- One one-dimensional periodic problem, `N=64`, `nu=0.01`, 50 validation
  trajectories, CPU, single-threaded BLAS.
- Exploratory throughout; the aggregator refuses inputs recorded at any other
  architecture, so these numbers cannot be pooled with the oversized
  architecture's.

## Where this leaves the Burgers lane

The reduction is adopted: any further Burgers work should use the
33-parameter pointwise-flux anchored generator, not the screen's
1,313-parameter five-point one.

The Burgers conclusion is now, in full: after matching everything and removing
the architecture's dead weight, time homogeneity buys a large and robust
structural property — exact in-range composition and exact unseen-lag
prediction consistency — and buys nothing in predictive accuracy, whose
sub-material sign depends on how much capacity the model has.

The corresponding questions for Fisher--KPP are both still open, and neither
can be answered from this repository: its architecture size was never ablated,
and its composition defects were never decomposed into a conditioning
artifact and an in-range term.  Both are cheap and checkpoint-only, but need
the SCNet checkpoints.

Machine-readable per-seed summaries, both checkpoints per seed, input digests,
the pooled aggregate, and the launcher are stored under
`results/20260905-burgers-minimal-generator-ab-3seed-cpu-r1/`.
