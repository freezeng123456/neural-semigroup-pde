# Burgers query-conditioning attribution: completed exploratory results

Date: 2026-09-05

The frozen decomposition in
`BURGERS_QUERY_CONDITIONING_ATTRIBUTION_PROTOCOL.md` is complete on all six
screen checkpoints.  Nothing was trained, selected, or tuned.  All values are
`exploratory=true` and `do_not_use_for_formal=true`.

## Outcome

Both frozen rules that could fire, fired.  The screen's headline composition
defect is almost entirely an artifact of out-of-range conditioning, **and** a
genuine in-range structural defect exists underneath it.

Geometric means over three seeds, on the 16 validation initial conditions the
screen itself used.

| Path | Autonomous | Query-time | Query-time over integrator floor | Verdict |
|---|---:|---:|---:|---|
| 1. Frozen semantics, direct path conditioned at the horizon | `4.717557e-8` | `1.621747e-1` | `3.44e+6` | `structural`, 12/12 |
| 2. Matched conditioning, both paths on one vector field | `4.717557e-8` | `4.414600e-8` | `0.936` | `collapsed`, 12/12 |
| 3. In-range cross-lag composition at equal work | `0.0` exactly | `5.751797e-3` | `1.22e+5` | `structural`, 6/6 |

The pooled conclusion recorded by the aggregator is
`headline_defect_is_conditioning_artifact_in_range_defect_is_structural`.

Path 1 reproduced the committed screen composition values in all 24 cells to
every printed digit, and its pooled geometric mean `1.621747e-1` equals the
screen's `geometric_mean_query_time` exactly.  An independent rerun from a
clean work root, regenerating the uncommitted cache from `--data-seed 20260902`
to its recorded digest, reproduced the aggregate byte for byte.

## What collapsed

Under matched conditioning the query-time model's defect falls from
`1.621747e-1` to `4.414600e-8`, a factor of `2.722126e-7`, and lands *below*
the autonomous integrator floor at `0.936` of it.  Per seed and cell the ratio
to the floor ranges from `0.81` to `1.14`.

So when both paths present the same value to the control channel, the
query-time model is exactly as composition-consistent as the autonomous model.
The `3.4`-million-fold ratio reported by the screen measures how far the
learned flux extrapolates when its control input jumps from the trained range
`0.3125`--`1.25` to `5` or `10`, not how far the model is from having one
generator.  **That ratio must not be quoted as the structural effect size.**

## What survived

Path 3 composes the same model at two lags it was trained and evaluated on,
`0.04` and `0.08`, to the same horizon, with the substep count scaled so both
paths spend the same number of right-hand-side evaluations.  Every call is
conditioned on an in-range value, so no extrapolation is involved.

| Seed | Horizon `0.4` | Horizon `0.8` |
|---:|---:|---:|
| `31415` | `3.774746e-3` | `8.260738e-3` |
| `271828` | `4.038904e-3` | `9.147570e-3` |
| `161803` | `3.781151e-3` | `8.312305e-3` |

Against a composed-state RMS of `0.456987`, the pooled in-range defect
`5.751797e-3` is `1.2586%` of the state norm.  The autonomous model's value is
exactly `0.0` in all six cells.

The mechanism claim therefore stands on its own without the extrapolation
term: two in-range members of the query-time family, given identical time
steps and identical work, disagree by roughly one to two percent of the state
norm, while the autonomous model's two paths are the same trajectory.

## An unintended sharpening of path 3

Equalizing right-hand-side work between the two cross-lag paths forces an
identical substep size: `12` substeps per `0.04` and `24` per `0.08` are both
`300` substeps per unit time.  The two paths are therefore the *same* sequence
of Runge--Kutta steps whenever the vector field does not depend on the
requested lag.

The frozen protocol expected the autonomous paths to "agree to integrator
accuracy".  They agree exactly, to the last bit, which the aggregator now
enforces as an acceptance check.  Path 3 consequently carries no integrator
contribution at all and is a pure measurement of in-range control-channel
sensitivity — a cleaner instrument than the freeze anticipated.

## A frozen rule that was ill-posed

The protocol's fourth interpretation rule asked whether the autonomous path-2
defect is small relative to the autonomous path-1 defect.  That comparison is
degenerate: the autonomous model zeroes its control channel, so the two paths
are identical by construction and the two defects are bitwise equal.  The
evaluator's test suite now asserts exactly that identity.

The question the rule was reaching for — whether nonuniform Runge--Kutta
truncation is negligible here — is answered directly instead.  The autonomous
nonuniform defect is `4.717557e-8` against a composed-state RMS of `0.456987`,
a relative error of `1.0e-7`, and the uniform-substep cross-lag defect is
exactly zero.  Integrator error therefore sits about five orders of magnitude
below the in-range structural defect and cannot explain it.

## Robustness

Repeating the whole decomposition on all 50 validation initial conditions
rather than the screen's 16 changes nothing qualitatively: floor
`5.049943e-8`, path 1 `1.752685e-1`, path 2 `4.614504e-8` at `0.914` of the
floor, path 3 `6.278609e-3` at `1.3135%` of the state norm, and the same
unanimous verdicts.

## Consequences

- The Burgers structural claim is restated in terms of path 3.  The
  defensible number is an in-range composition defect of about `1.3%` of the
  state norm against exactly zero, not a ratio of `2.9e-7`.
- That number is architecture dependent but the conclusion is not.  Repeating
  the path-3 measurement on the 33-parameter generator recommended by
  `BURGERS_FLUX_ABLATION_RESULTS.md` gives `0.7456%` of the state norm at
  `9.01e+4` times the integrator floor, still structural in all six cells; see
  `BURGERS_MINIMAL_GENERATOR_AB_RESULTS.md`.  The mechanism therefore holds at
  two architectures forty times apart in parameter count.
- The accuracy conclusion of the screen is untouched.  The geometric-mean
  rollout-MSE ratio remains `0.9933`, still short of the `0.90` material
  threshold, so the Section 4 advance criterion remains unmet on two PDEs.
- The screen's own caveat is now resolved rather than open, and the mechanism
  result is robust to the most obvious methodological objection: that the
  defect was manufactured by asking the model for a duration it never saw.
- **The same restatement is now owed to Fisher--KPP.**
  `FISHER_FORMAL_CLOSURE_AND_MECHANISM_SCREEN.md` already reports a drop from
  `4.2889374e-05` to `4.4894191e-08` between normal and fixed-query-time
  conditioning at horizon `4.8` for seed `271828`, roughly `956`-fold and the
  same qualitative direction as path 2 here.  That lane never measured an
  in-range cross-lag path, so the in-range magnitude of the Fisher structural
  effect is currently unknown, and the Fisher composition-defect ratios in the
  novelty and phase-diagram results carry the same inflation risk.  Running
  this decomposition on the frozen Fisher checkpoints is checkpoint-only and
  cheap, but those checkpoints and the locked cache live on SCNet rather than
  in this repository.

Machine-readable per-seed values, both sample sets, the pooled aggregates,
input digests, and the launcher are stored under
`results/20260905-burgers-query-conditioning-attribution-fd27d7d-r1/`.
