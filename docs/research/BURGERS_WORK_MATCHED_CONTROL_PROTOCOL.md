# Work-matched flow versus direct-map protocol

Status: frozen exploratory protocol, written before observing any result at
matched work.

## Question

`BURGERS_DIRECT_MAP_CONTROL_RESULTS.md` left exactly one thing open.  Its
structural verdict was clean and compute-independent, but its accuracy verdict
was confounded: model C spent one flux evaluation per call against the flows'
forty-eight, trained in `1.6 s` against `26.8 s`, and was `6.1` times worse on
the one-step selection metric.  A `30%` rollout advantage under that asymmetry
is not evidence about autonomy.

This lane matches the work and asks whether the accuracy advantage survives.

## Why the flows must come down rather than C going up

A one-shot direct map cannot be given more flux evaluations per call without
becoming a flow, unless it is also given more parameters.  Iterating a
lag-conditioned increment `K` times at `tau/K` *is* the query-conditioned
flow; adding per-layer weights breaks the parameter match that makes the
comparison meaningful.  The only parameter-matched way to equalise work is
therefore to reduce the flow's integrator budget.

Explicit Euler with one substep costs exactly one flux evaluation per call, so

$A'(u,\tau)=u+\tau F_\theta(u)$, $\qquad B'(u,\tau)=u+\tau F_\theta(u;\tau)$

are the work-matched counterparts of C.  All three then have `33` parameters
and one flux evaluation per call.

This is a real cost to the flow models and it is expected to show.  A single
Euler step cannot represent the exact flow over the requested duration, so the
learned field must absorb a discretisation compromise across the training lags
`0.025`--`0.1`.  Measuring that cost honestly is the point.

## The structural question changes shape at matched work

At the deployed budget the two in-range cross-lag paths of a flow use
different substep sizes, `dt=0.04` against `dt=0.08`, so a flow's cross-lag
defect is no longer exactly zero.  Its exactness in
`BURGERS_QUERY_CONDITIONING_ATTRIBUTION_RESULTS.md` came from the equal-work
convention there, which happens to equalise `dt` as well.

What still separates the formulations at matched work is **refinability**.  A
flow model defines a vector field, so its composition defect is integrator
error and shrinks as the evaluation-time substep count grows toward the exact
flow, which composes exactly.  A direct map defines no vector field, so it has
no refinement axis at all and its defect is irreducible.

The lane therefore measures a refinement sweep, not a single number.

## Frozen design

- **Trained fresh:** `A'` and `B'`, the adopted minimal generator integrated by
  explicit Euler with one substep.  Three seeds `31415`, `271828`, `161803`.
  `B'` is initialised from `A'`'s parameter tensors, as in every earlier
  paired lane.
- **Loaded frozen:** model C, the committed checkpoints of
  `results/20260905-burgers-direct-map-control-3seed-cpu-r1/`.  Never
  retrained or written to.
- **Everything else at the screen's frozen values:** the immutable cache
  regenerated from `--data-seed 20260902` with digest
  `30c8443e37fa9b0d97e2b964699e8c0044f8f637232acf29abf915f19ac2de8d`, 1,000
  training pairs, 50 validation trajectories, training lags `0.025`--`0.1`,
  unseen evaluation lags `0.04` and `0.08`, horizons `0.4` and `0.8`, 100
  epochs, batch size 64, AdamW `1e-3` with `1e-5` weight decay, validation
  every ten epochs, and checkpoint selection by mean one-step validation MSE
  on the four training lags only.

## Endpoints

Primary, at exactly one flux evaluation per call for all three models:
geometric-mean rollout-MSE ratios `A'/C` and `B'/C` over the four
unseen-lag/horizon cells, pooled over three seeds, against the project's
`0.90` material threshold.

Structural:

1. the in-range cross-lag defect at the deployed budget for all three models;
2. an evaluation-time refinement sweep over `1, 2, 4, 8, 16` substeps per call
   for `A'` and `B'`, reported with its flux-evaluation count, against C's
   single irreducible value;
3. the equal-substep in-range cross-lag defect for `A'`, which must be exactly
   zero because its field is autonomous.

Also reported: the one-step selection metric, spatial-mean drift,
quadratic-energy change, training wall clock, and flux evaluations per call
for every measurement.

## Frozen decision rules

**Accuracy.** `flow_retains_accuracy_advantage_at_matched_work` when the
pooled `A'/C` ratio is at most `0.90` and at least two of three seeds agree;
`flow_accuracy_advantage_does_not_survive_work_matching` otherwise.

**Refinability.** `flow_defect_refines_direct_map_defect_does_not` when all
three of the following hold in every seed:

- `A'` and `B'` have a monotonically decreasing cross-lag defect across the
  sweep `1, 2, 4, 8, 16`;
- each flow's defect at `16` substeps is at least `10` times smaller than at
  `1` substep;
- C's defect is at least `100` times each flow's defect at `16` substeps.

Otherwise `refinement_does_not_separate_the_formulations`.

Both outcomes of both rules are declared results in advance.  If the accuracy
advantage does not survive, the honest Burgers conclusion becomes that the
flow formulation buys structure and refinability and costs accuracy per unit
of compute, and the project must stop quoting the `0.6992` figure.

This lane is exploratory, cannot be pooled with the locked Fisher decision or
with the unmatched-work control, and does not alter any completed lane.  Its
aggregator refuses inputs recorded at any other architecture or at any other
flux-evaluation budget.
