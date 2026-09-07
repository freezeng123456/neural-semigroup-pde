# Minimal Burgers generator A/B protocol

Status: frozen exploratory protocol, written before observing any result on
the reduced generator.

## Question

`BURGERS_FLUX_ABLATION_RESULTS.md` recommends replacing the screen's
1,313-parameter flux generator with `r0_h8_l1_anchor`: a pointwise flux, one
hidden layer of width 8, 33 parameters, about half the rollout error.

Both of the Burgers lane's headline numbers were measured on the architecture
that ablation identified as the worst in its own grid:

- the accuracy endpoint, geometric-mean `MSE(A)/MSE(B)` of `0.9933176` over 12
  cells, from `BURGERS_MATCHED_SEMIGROUP_RESULTS.md`;
- the structural endpoint, an in-range cross-lag composition defect of
  `5.751797e-3`, which is `1.2586%` of the composed-state RMS, against exactly
  zero for the autonomous model, from
  `BURGERS_QUERY_CONDITIONING_ATTRIBUTION_RESULTS.md`.

A generator forty times smaller and twice as accurate can move either number.
In particular, with patch radius zero the flux network reads only two inputs,
the local state and the control channel, so it may simply learn to ignore the
control channel; that would remove the structural effect rather than confirm
it.  This protocol asks whether both conclusions survive the reduction, which
is the precondition for adopting it.

## Frozen design

One paired A/B comparison at the recommended architecture, nothing else
changed.

- **Architecture:** patch radius `0`, hidden width `8`, one hidden layer, fixed
  viscous anchor.  Model A zeroes the control channel; model B feeds the
  requested lag through it.  The control column exists in both, so both models
  have exactly `33` parameters, and B is initialized from A's parameter
  tensors as in the screen.
- **Data:** the screen's immutable cache, regenerated from `--data-seed
  20260902` and required to hash to
  `30c8443e37fa9b0d97e2b964699e8c0044f8f637232acf29abf915f19ac2de8d`.
- **Seeds:** `31415`, `271828`, `161803`.
- **Everything else at the screen's frozen values:** training lags
  `0.025`--`0.1`, unseen evaluation lags `0.04` and `0.08`, horizons `0.4` and
  `0.8`, 100 epochs, batch size 64, AdamW `1e-3` with `1e-5` weight decay,
  validation every ten epochs, 12 RK4 substeps per call, and checkpoint
  selection by mean one-step validation MSE on the four training lags only.
- Inside a seed the two paired models train sequentially so they never contend
  for the same allocation.

## Endpoints

Primary: geometric-mean rollout-MSE ratio `A/B` over the four
unseen-lag/horizon cells, pooled over the three seeds, against the project's
`0.90` material threshold.

Structural: the same three-path decomposition already frozen in
`BURGERS_QUERY_CONDITIONING_ATTRIBUTION_PROTOCOL.md`, reusing that evaluator's
measurement function unchanged — frozen semantics, matched conditioning, and
in-range cross-lag at equal work — plus spatial-mean drift and quadratic-energy
change.

Two invariants are asserted rather than measured: A and B must report the same
parameter count, and A's in-range cross-lag defect must be exactly zero,
because equal work forces an identical substep size and A's control channel is
zeroed.

## Frozen decision rules

**Accuracy.** `accuracy_conclusion_unchanged` when the pooled ratio still
exceeds `0.90`, or meets it in fewer than two of three seeds.
`accuracy_conclusion_changed` when the pooled ratio is at most `0.90` and at
least two of three seeds agree, which is the screen's own advance criterion.

**Structure.** Using the bands already frozen for the attribution lane, with
the integrator floor defined as A's matched-conditioning defect:
`structure_conclusion_unchanged` when B's in-range cross-lag defect is
`structural`, meaning at least `100` times the floor, in all three seeds;
`structure_conclusion_changed` otherwise.

**Adoption.** The reduction is adopted only when both conclusions are
unchanged.  If the accuracy conclusion flips, the reduced generator supports a
claim the oversized one did not and the Burgers primary endpoint must be
rewritten around it.  If the structural conclusion flips, the reduced
generator does not exhibit the mechanism at all and the reduction must not be
applied to any lane that relies on it.  Either flip is a result, not a
failure; neither may be resolved by trying a third architecture.

This lane is exploratory, cannot be pooled with the locked Fisher decision or
with the oversized-architecture attribution results, and does not alter the
completed screen.  Its aggregator refuses any input not recorded at this
architecture.
