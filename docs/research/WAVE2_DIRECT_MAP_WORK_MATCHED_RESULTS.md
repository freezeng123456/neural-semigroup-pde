# Wave 2 direct-map control at matched work: completed exploratory results

Date: 2026-09-07

The frozen lane in `WAVE2_DIRECT_MAP_WORK_MATCHED_PROTOCOL.md` is complete.
All four frozen rules fired **positively**, on every boundary family.  This
is the first accuracy claim in the project that has survived a work-matched
direct-map control.  All values are `exploratory=true` and
`do_not_use_for_formal=true`.

## Outcome

On the Wave 2 reaction--diffusion PDE, at one network evaluation per call
and 1,249 parameters for every model:

- **`flow_has_material_accuracy_advantage_over_direct_map`** in all three
  families, `3/3` seeds each.  Pooled `MSE(A')/MSE(C)` is `0.0986`
  (Dirichlet), `0.0298` (Neumann), and `0.0209` (Robin).
- **`autonomous_defect_is_refinable`.**  `A'`'s in-range cross-lag defect
  shrinks by `438`--`648` from 1 to 128 Euler substeps, against a threshold
  of `64`.  Equal-substep composition is exactly `0` in every seed.
- **`query_time_has_irreducible_defect`.**  `B'`'s corresponding gain is
  `1.98`--`2.23`, below `8`.  After two substeps the series is already
  flat.
- **`structural_ordering_is_specific_to_autonomy`.**  `C`'s deployed
  defect exceeds `A'` at 128 substeps by `1,764`--`2,239`, against a
  threshold of `16`.

Burgers at matched work retracted the accuracy figure and left only the
structural claim.  The same two questions, asked on the PDE where Wave 2
had an A-versus-B accuracy number, keep both halves.

## Why this lane existed

Wave 2's hard-boundary long-rollout ratios `0.125`, `0.110`, and `0.089`
were A versus B: two RK4 flows.  The novelty audit's indispensable
comparison is A versus C.  That control had been run only on Burgers, and
the unmatched `0.699` retracted to `1.008` once work was matched.  This
lane therefore trained `A'`, `B'`, and `C` at one evaluation per call
from the first cell, and it used the corrected refinability rule that the
Burgers work-matched note could not adopt.

`A` and `B` were retrained at Wave 2's RK4-8 budget as a descriptive
check.  Their long-rollout geometric means reproduce Wave 2 to four
significant figures, so the new control is not an accidental change of
PDE, data, or architecture.

| Family | Wave 2 `MSE(A)/MSE(B)` | This lane, RK4-8 |
|---|---:|---:|
| Homogeneous Neumann | `0.124960` | `0.124959` |
| Inhomogeneous Dirichlet | `0.109899` | `0.109903` |
| Robin | `0.088586` | `0.088580` |

## Accuracy at matched work

Primary readout: geometric-mean long-rollout MSE over
\((\tau,H)\in\{0.03,0.06\}\times\{0.24,0.48\}\), three seeds, one family
at a time.  Ratios below one favour the flow.  The material threshold is
`0.90`.

| Family | `MSE(A')/MSE(C)` | Seeds \(\le 0.90\) | `MSE(B')/MSE(C)` | `MSE(A')/MSE(B')` |
|---|---:|---:|---:|---:|
| Inhomogeneous Dirichlet | `0.0986` | `3/3` | `0.1156` | `0.853` |
| Homogeneous Neumann | `0.0298` | `3/3` | `0.0343` | `0.868` |
| Robin | `0.0209` | `3/3` | `0.0262` | `0.795` |

Absolute long-rollout geometric means:

| Family | `A'` | `B'` | `C` | `A` (RK4-8) |
|---|---:|---:|---:|---:|
| Dirichlet | `2.72e-3` | `3.19e-3` | `2.76e-2` | `1.30e-5` |
| Neumann | `1.89e-3` | `2.17e-3` | `6.33e-2` | `1.44e-5` |
| Robin | `1.47e-3` | `1.85e-3` | `7.07e-2` | `1.45e-5` |

Training wall clock is matched among the three one-evaluation models:
`2.7 s`, `2.7 s`, and `2.3 s` per model on an H20.  The RK4-8 flows take
about `80`--`93 s` and are not part of the accuracy verdict.

Unmatched `MSE(A)/MSE(C)` is `2e-4` to `5e-4`.  That number is
descriptive only.  The protocol forbids reading a win from it; the
matched ratios already suffice.

## What the one-step metric does and does not say

Checkpoint selection sees only one-step validation MSE on the four
training lags.  On Neumann those selected values are `4.15e-4` (`A'`),
`4.07e-4` (`B'`), and `7.94e-4` (`C`).  `C` is less than twice `A'` on
the quantity the optimiser is judged by, and about thirty-three times
worse on the long-rollout geometric mean.  The accuracy gap is therefore
not a failure of `C` to fit a single step.  It is a failure of those
fitted steps to compose.

That is the same mechanism the structural endpoints measure, now visible
in prediction error.

## Corrected refinability

In-range cross-lag defect at horizon `0.16`, four steps of `0.04` against
two steps of `0.08`.  Geometric means across the three seeds.

| Substeps | Dirichlet `A'` | Dirichlet `B'` | Neumann `A'` | Neumann `B'` | Robin `A'` | Robin `B'` |
|---|---:|---:|---:|---:|---:|---:|
| `1` | `9.59e-3` | `1.27e-2` | `6.50e-3` | `9.26e-3` | `6.79e-3` | `9.31e-3` |
| `2` | `1.04e-3` | `5.83e-3` | `1.06e-3` | `4.90e-3` | `9.76e-4` | `4.95e-3` |
| `4` | `4.97e-4` | `5.72e-3` | `5.01e-4` | `4.67e-3` | `4.60e-4` | `4.78e-3` |
| `16` | `1.20e-4` | `5.69e-3` | `1.20e-4` | `4.57e-3` | `1.10e-4` | `4.71e-3` |
| `128` | `1.48e-5` | `5.69e-3` | `1.48e-5` | `4.55e-3` | `1.36e-5` | `4.70e-3` |
| `C` (no axis) | `2.61e-2` | | `2.97e-2` | | `3.04e-2` | |

From two substeps onward `A'` halves at every doubling, which is exactly
first-order Euler.  `B'` drops once and then stops: the refinable part is
integrator error, the floor is the disagreement between the fields
conditioned on `0.04` and `0.08`.  `C` has no sweep.

The Burgers work-matched rule asked `B'` to refine by `10` and demanded a
`100`-fold separation at 16 substeps.  Both demands were specification
errors.  The corrected rule, written before this lane ran, is the one
that fired here.

## Boundary gate

Hard enforcement is exact on every trained model, including `C`.  Wave
2's endpoint-stress states, advanced with \(\tau=0.04\) to \(H=0.16\),
have maximum boundary residual `0.0` in float32 for all five labels, all
three families, and all three seeds.  Spatial admissibility and temporal
formulation remain independent interventions.

## How to read `C`, and what this does not claim

`C` is the protocol's residual operator
\(C(u,\tau)=P_B(u+N_\theta(u,\tau))\).  The stencil MLP is the Wave 2
vector-field net, including the duration feature \(\tau/0.08\), but the
output is an unscaled state residual.  That is a deliberate distinction
from \(B'(u,\tau)=u+\tau F_\theta(u;\tau)\).  It is also a weaker
inductive bias than Burgers' physics-split map
\(e^{\tau\nu D_2}u+\tau(-D_0 f_\theta)\), which keeps a free factor of
\(\tau\) on the nonlinear increment.

`B'` beats `C` by about the same factor that `A'` does.  Autonomy is
therefore not the only difference between a field and this residual map.
Two comparisons stay meaningful:

1. `A'` versus `C` is the frozen primary readout, and it is material.
2. `A'` versus `B'` is `0.795`--`0.868`, which would itself meet the
   `0.90` threshold in every family.  Autonomy still helps once both
   models are fields with a free factor of \(\tau\`.

A Burgers-style physics-split `C` on this PDE — exact boundary-aware heat
semigroup plus one residual reaction — is a different control and was
not run.  It cannot be substituted after the fact.  The claim supported
here is narrower than "any direct map loses": it is that the Wave 2
autonomous flow, at matched parameters and matched work, predicts
materially better than this residual time-conditioned operator, and that
its composition defect is refinable integrator error.

The result is for one one-dimensional cubic reaction--diffusion equation,
hard enforcement, and the Wave 2 stencil.  It does not reopen the locked
Fisher decision, and it does not restore an accuracy claim on Burgers.

## Provenance

- protocol: `docs/research/WAVE2_DIRECT_MAP_WORK_MATCHED_PROTOCOL.md`
- source branch tip before execution: `097a466`
- caches, three-seed cells, checkpoints, and the aggregator output:
  `experiments/results/20260907-wave2-direct-map-work-matched-3seed-h20-r1/`
- cache digests:
  Dirichlet `295534fb9f8637285ff15bdb26980b5df6f473a5dd78aaefe707b60f706e18b4`,
  Neumann `3d55869bb31e7810826705d4c7d37ae3df4483884365b1995efedac3262fb6b7`,
  Robin `20c1777c3f4b9ab35ef0183f6926bfdcbeabf635621fbbc65271268f6ba47e1d`
- device: NVIDIA H20, `torch==2.5.1+cu121`
- aggregator verdicts recorded in `aggregate.json`

The aggregator refused smoke cells, unmatched Euler budgets, and any
architecture other than the 1,249-parameter Wave 2 stencil.
