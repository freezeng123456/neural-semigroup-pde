# Burgers direct-map control: completed exploratory results

Date: 2026-09-05

The frozen lane in `BURGERS_DIRECT_MAP_CONTROL_PROTOCOL.md` is complete.  This
is the A-versus-C comparison that `CURRENT_METHOD_NOVELTY_PRIOR_ART.md` calls
indispensable and that the project had never run.  All values are
`exploratory=true` and `do_not_use_for_formal=true`.

## Outcome

Both frozen rules fired.  One verdict is clean and one is confounded, and they
must be read very differently.

- **`structural_property_is_specific_to_autonomy`** — clean, and independent of
  compute.
- **`autonomy_has_material_accuracy_advantage_over_direct_map`** — fired, but
  **not adoptable**: model C ran with `1/48` of the flux evaluations and
  `1/17` of the wall clock.  See the confound section below.

**Retracted.**  `BURGERS_WORK_MATCHED_CONTROL_RESULTS.md` repeated the
comparison at one flux evaluation per call for all three models and obtained
`MSE(A')/MSE(C) = 1.0084` with no seed meeting the threshold.  The `0.6992`
figure below was entirely a compute artifact.  The refusal to adopt it was
correct; it must not be quoted.

## The structural result: a monotone ordering across three temporal formulations

The three models share 33 parameters, the same flux network, the same periodic
divergence, the same fixed viscous anchor, the same data, and the same
training and selection rules.  They differ only in how much freedom the
requested duration has to change the map.

| | Freedom given to the requested lag | In-range cross-lag defect, relative to state RMS | Unseen-lag rollout-MSE spread |
|---|---|---:|---:|
| A, autonomous flow | none | `0` exactly | `2.978145e-7` |
| B, query-conditioned flow | the vector field reads it | `0.7456%` | `7.963157e-2` |
| C, direct time-conditioned map | the whole increment is a learned function of it | `2.8013%` | `7.441013e-1` |

C's in-range cross-lag defect is `1.267133e-2`, which is `341,603` times the
autonomous integrator floor, and it is judged `structural` in all six cells.
Its per-seed values are extremely tight: `8.12e-3` to `8.22e-3` at horizon
`0.4` and `1.94e-2` to `1.99e-2` at horizon `0.8`.

The ordering spans roughly seven orders of magnitude on the prediction-level
endpoint and it is monotone in exactly the property the lane set out to test.
**This ordering cannot be a compute artifact.**  Giving C more evaluations per
call would not make its `tau=0.04` and `tau=0.08` maps iterates of one flow;
nothing in a direct operator constrains them to be.  Exact in-range
composition and exact unseen-lag prediction consistency are therefore
properties of the autonomous flow formulation that neither a
query-conditioned flow nor an ordinary direct time-conditioned map shares.

This is the structural half of what the novelty audit asked for, and it is the
first result in the project that separates *autonomy* from *continuous-time
integration*: B and C are both non-autonomous, but B integrates a field and C
does not, and they sit an order of magnitude apart.

## The accuracy result is confounded by compute and is not adopted

The pooled ratio is `MSE(A)/MSE(C) = 0.6992` with all three seeds inside a
`0.0009` band (`0.6996`, `0.6994`, `0.6987`), and `MSE(B)/MSE(C) = 0.6780`.
By the letter of the frozen rule that is a material advantage in `3/3` seeds.
It should not be used, for three reasons recorded together:

| | A | B | C |
|---|---:|---:|---:|
| Flux evaluations per call | `48` | `48` | `1` |
| Training wall clock per model | `26.8 s` | `26.6 s` | `1.6 s` |
| One-step selection metric | `2.493336e-6` | `1.528587e-6` | `1.524238e-5` |

C is `6.1` times worse than A on the one-step metric, the only quantity
checkpoint selection sees, while spending a small fraction of the compute.  A
`30%` rollout advantage under that asymmetry is fully consistent with simply
spending more compute per call and says nothing about autonomy.

The frozen protocol anticipated the asymmetry but only in one direction: it
declared that because C is cheaper, a *tie* on accuracy would be a strong
statement against the flow formulation.  It did not license reading a flow
*win* as clean, and this note does not.  The novelty audit is equally explicit
that its conditional structural claim requires the A-versus-C comparison to be
**at matched work**.

The accuracy half of the audit's criterion therefore remains open.

## Cell structure, which further limits the accuracy reading

| Cell | A | B | C | `A/C` |
|---|---:|---:|---:|---:|
| `tau=0.04, T=0.4` | `3.396826e-4` | `3.329083e-4` | `4.839501e-4` | `0.7019` |
| `tau=0.04, T=0.8` | `1.389470e-2` | `1.387699e-2` | `1.437647e-2` | `0.9665` |
| `tau=0.08, T=0.4` | `3.396826e-4` | `3.092166e-4` | `8.413643e-4` | `0.4037` |
| `tau=0.08, T=0.8` | `1.389470e-2` | `1.378001e-2` | `1.591768e-2` | `0.8729` |

The advantage is concentrated at the shallow composition depth and nearly
disappears at horizon `0.8`, where all three models sit near the same
`1.4e-2` error level.  The pooled `0.6992` is therefore driven by the two
short-horizon cells, and at the long horizon the three formulations are
close.  Note also that A's two lag columns are identical to seven digits,
which is the autonomy signature again, while C's differ by a factor of `1.7`.

## Other endpoints

Spatial-mean drift is at most `7.870e-8` for A, `6.892e-8` for B and
`4.354e-8` for C, so all three stay in divergence form; C's exact viscous
exponential preserves the constant Fourier mode by construction.

Mean quadratic-energy change is at most `-5.201e-3` for A, `-5.216e-3` for B
and `-2.023e-3` for C, against reference values of `-4.769e-3` at horizon
`0.4` and `-9.953e-3` at horizon `0.8`.  C under-dissipates and the flows
over-dissipate at the long horizon, but the reference's own dealiasing problem
recorded in `BURGERS_FLUX_ABLATION_RESULTS.md` limits this to a sign-level
reading.

C's direct-versus-composed defect is `1.10e-1` to `2.46e-1`.  That number
carries the same extrapolation caveat as the flow lane's path 1: the direct
call requests `tau=0.4` or `0.8`, far outside the trained range
`0.025`--`0.1`.  It is reported for completeness and is not part of any
verdict; the in-range cross-lag endpoint is the one the decision rule uses.

## Provenance

- Models A and B are the committed frozen checkpoints of
  `results/20260905-burgers-minimal-generator-ab-3seed-cpu-r1/`, loaded
  read-only.  The launcher hashes all six before and after execution and
  requires them to be byte-identical.
- The immutable cache digest
  `30c8443e37fa9b0d97e2b964699e8c0044f8f637232acf29abf915f19ac2de8d` is
  identical before and after execution and equal to the screen's.
- C is asserted to have exactly `33` parameters, and its test suite asserts
  that C's flux weights under a given seed are identical to A's, that both of
  its terms preserve the spatial mean, that its linear factor is an exact
  semigroup, and that its two in-range lag paths genuinely disagree.
- The aggregator refuses any summary not recorded at this architecture.

## What is now established, and what the next step must be

Established, compute-independently: among three temporal formulations at
identical parameter budget and identical physics, exact in-range composition
consistency and exact unseen-lag prediction consistency belong to the
autonomous flow alone, and the degradation is monotone in how much freedom the
requested duration is given.  A direct time-conditioned map is the worst of
the three by a wide margin on both endpoints.

Not established: that the flow formulation is more accurate.  Every accuracy
comparison in this lane is unmatched in work by a factor of `48` in flux
evaluations.

Design 2 of the work-matched follow-up — retraining the flows at a reduced
substep budget — has been run in
`BURGERS_WORK_MATCHED_CONTROL_RESULTS.md`.  At one flux evaluation per call
the three formulations are equally accurate to within one percent, so the
accuracy half of the novelty audit's criterion is now answered in the
negative: **the project has no evidence that the flow formulation predicts
better than a direct operator.**

Design 1, giving C the same `48` evaluations as a deep residual operator with
per-layer weights, remains unrun.  It answers the complementary question of
whether a *larger* direct operator can match the flow, and it necessarily
breaks the parameter match.

Machine-readable per-seed summaries, C's checkpoints, input and frozen-
checkpoint digests, the pooled aggregate, and the launcher are stored under
`results/20260905-burgers-direct-map-control-3seed-cpu-r1/`.
