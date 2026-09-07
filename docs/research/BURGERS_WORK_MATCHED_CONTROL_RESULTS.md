# Work-matched flow versus direct map: completed exploratory results

Date: 2026-09-05

The frozen lane in `BURGERS_WORK_MATCHED_CONTROL_PROTOCOL.md` is complete.
Both frozen rules fired **negatively**, and one of them retracts a result this
project reported only hours earlier.  All values are `exploratory=true` and
`do_not_use_for_formal=true`.

## Outcome

- **`flow_accuracy_advantage_does_not_survive_work_matching`.**  At one flux
  evaluation per call for all three models, the pooled ratios are
  `MSE(A')/MSE(C) = 1.0084` and `MSE(B')/MSE(C) = 1.0013`.  No seed meets the
  `0.90` threshold; the flow is marginally *worse*.
- **`refinement_does_not_separate_the_formulations`.**  The rule failed, but
  for two specification errors of mine rather than because the underlying
  phenomenon is absent.  See the two sections below, kept separate on purpose.

## The accuracy advantage was compute

| | Unmatched work | Matched work |
|---|---:|---:|
| Flow flux evaluations per call | `48` | `1` |
| `MSE(A)/MSE(C)` | `0.6992` | `1.0084` |
| `MSE(B)/MSE(C)` | `0.6780` | `1.0013` |
| Seeds meeting `0.90` | `3/3` | `0/3` |

Per seed the matched ratios are `1.0070`, `1.0103`, `1.0079`, so the tie is as
tight as the earlier advantage was.  The one-step selection metric ties too:
`1.586e-5` for `A'`, `1.523e-5` for `B'`, against C's `1.524e-5`.

`BURGERS_DIRECT_MAP_CONTROL_RESULTS.md` refused to adopt its own `0.6992`
verdict and predicted exactly this, so the refusal was correct and the figure
is now formally retracted.  **The project has no evidence that the flow
formulation predicts better than a direct time-conditioned map.**  At matched
parameters and matched work, on this PDE, all three temporal formulations are
equally accurate to within one percent.

What the flow buys with its 48 evaluations is real but is a compute story: the
RK4 flow reaches a one-step metric of `2.493e-6` against `1.586e-5` for the
Euler flow at one evaluation, a factor of `6.4` for a factor of `48` in work.

## The descriptive refinement structure, which is clean

This section reports what was measured.  It is **not** a verdict; the frozen
rule that was supposed to test it is discussed next.

Pooled in-range cross-lag defect against substeps per call:

| Substeps | `1` | `2` | `4` | `8` | `16` |
|---|---:|---:|---:|---:|---:|
| `A'`, autonomous flow | `1.0457e-2` | `4.6422e-3` | `2.1852e-3` | `1.0607e-3` | `5.2269e-4` |
| `B'`, query-conditioned flow | `1.2498e-2` | `6.3813e-3` | `4.1668e-3` | `3.3990e-3` | `3.1421e-3` |
| `C`, direct map | `1.2671e-2` | — | — | — | — |

Ratio per doubling:

| | `1→2` | `2→4` | `4→8` | `8→16` |
|---|---:|---:|---:|---:|
| `A'` | `2.253` | `2.124` | `2.060` | `2.029` |
| `B'` | `1.959` | `1.531` | `1.226` | `1.082` |

The three formulations behave in three distinct ways:

- `A'` shows clean first-order convergence toward zero.  The ratio approaches
  `2` per doubling, which is exactly what explicit Euler should give, so its
  cross-lag defect is integrator error and nothing else.  Its exact flow
  composes exactly, and the artifact separately confirms that at equal
  substeps the defect is `0.0` to the last bit.
- `B'` has a refinable part and an irreducible one.  Its ratio decays toward
  `1` and its successive decrements shrink by about a factor of three, so it
  plateaus near `3.0e-3`.  This is the expected behaviour: refining substeps
  converges to the exact flows of two *different* vector fields, one
  conditioned on `0.04` and one on `0.08`, and those flows still disagree.
- `C` has no refinement axis at all.  Its `1.2671e-2` is a single number.

## The frozen rule was mis-specified, and fired negative

The rule required, in every seed and cell, that both flow models refine by at
least `10` between `1` and `16` substeps, and that C's defect exceed each
flow's `16`-substep defect by at least `100`.  Observed:

| Check | Result |
|---|---|
| Monotone in every cell | **pass** |
| Refinement gain at least `10` | **fail** — `A'` gains `20.0`, `B'` only `3.98` |
| C separated by at least `100` | **fail** — `C/A'@16 = 24.2`, `C/B'@16 = 4.0` |

Both failures are specification errors, not model behaviour:

1. **Requiring `B'` to refine was wrong.**  `B'` is not autonomous, so its
   cross-lag defect is not supposed to vanish under refinement; it has an
   irreducible structural component by construction, which is the very thing
   the earlier lanes measured.  The rule conflated the refinable integrator
   part with the irreducible structural part and then demanded that the
   structural part disappear.
2. **The separation factor was inconsistent with a first-order integrator.**
   `A'` is explicit Euler, so `16` substeps buy only about `20`.  Reaching a
   `100`-fold separation needs roughly `64` to `128` substeps.  Freezing a
   `100` factor at a sweep that stops at `16` was arithmetically incompatible
   with the integrator being tested.

The rule therefore cannot be read as evidence either way, and this note does
not reinterpret it.  The descriptive structure above is reported as
descriptive.  Establishing the refinability separation as a *result* requires a
new preregistration with a corrected rule: apply the refinement requirement to
the autonomous model only, extend the sweep to at least `128` substeps, and
set the separation factor from the integrator's expected order rather than
picking a round number.  That is a fresh lane, not a reinterpretation of this
one.

## Other endpoints

Both flow models preserve the spatial mean to float32 roundoff at the Euler
budget, and their training times are `1.56 s` and `1.53 s` per model against
C's `1.6 s`, so the wall clock is matched as well as the evaluation count.

## Provenance

- Model C is the committed frozen checkpoint of
  `results/20260905-burgers-direct-map-control-3seed-cpu-r1/`, loaded
  read-only; the launcher hashes all three before and after execution and
  requires them byte-identical.
- The immutable cache digest
  `30c8443e37fa9b0d97e2b964699e8c0044f8f637232acf29abf915f19ac2de8d` is
  unchanged across execution and equal to the screen's.
- The runner asserts that all three models carry exactly `33` parameters and
  that the flux-evaluation budget is `1` for each.  A test asserts by call
  counting that one Euler substep really is one field evaluation and that the
  deployed map equals `u + tau F(u)` exactly.
- The aggregator rejects any summary at another architecture or at an
  unmatched budget, and rejects a nonzero autonomous equal-substep defect.

## Where this leaves the Burgers lane

The honest Burgers conclusion, after the screen, the attribution lane, the
ablation, the adopted reduction, the direct-map control and this one:

> At matched parameters and matched work, the autonomous flow, the
> query-conditioned flow and the direct time-conditioned map are equally
> accurate on this problem to within one percent.  What time homogeneity buys
> is structural: with a sufficient integrator budget the autonomous flow's
> composition and unseen-lag prediction defects are exactly zero, and at any
> budget they are pure integrator error that refines toward zero, whereas the
> other two formulations have defects bounded below.

Nothing in the Burgers lane supports an accuracy claim of any kind, in either
direction.  Two frozen accuracy verdicts have now fired in opposite
directions, and the one that favoured the flow has been retracted.

The next steps, in order:

1. the corrected refinability lane described above, which is the only thing
   that would turn the clean descriptive structure into a result;
2. the sharp discrete one-sided estimate identified in
   `BURGERS_CERTIFIED_CONSTANTS.md`, worth about three orders of magnitude in
   the certified bound;
3. the two Fisher gap-fills, which need the SCNet checkpoints.

Machine-readable per-seed summaries, both Euler checkpoints per seed, digests,
the refinement sweep, and the pooled aggregate are stored under
`results/20260905-burgers-work-matched-control-3seed-cpu-r1/`.
