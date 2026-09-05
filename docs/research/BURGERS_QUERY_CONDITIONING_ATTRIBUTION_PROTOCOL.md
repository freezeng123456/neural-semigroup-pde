# Burgers query-conditioning attribution protocol

Status: frozen exploratory checkpoint-only protocol, written before observing
any of the decomposition results below.

## Question

The completed matched Burgers screen reported a geometric-mean equal-work
composition defect of `4.72e-8` for the autonomous model and `1.62e-1` for the
query-time model.  `BURGERS_MATCHED_SEMIGROUP_RESULTS.md` records why that
second number cannot be read at face value: in the frozen equal-work
definition the direct path feeds the whole horizon into the control channel,
which the model normalizes by `max(test_taus)=0.08`, so horizons `0.4` and
`0.8` present control inputs of `5` and `10` while training only ever
presented `0.3125`--`1.25`.

The reported defect therefore mixes two different things:

1. genuine query-time inhomogeneity, meaning that maps obtained at different
   requested lags are not iterates of one shared generator; and
2. extrapolation of the control channel far outside its training range.

This protocol separates them without training, tuning, checkpoint selection,
or any new model.  It is the Burgers analogue of the fixed-query-time control
that the Fisher lane already ran in
`FISHER_FORMAL_CLOSURE_AND_MECHANISM_SCREEN.md`.

## Frozen inputs

Use the six checkpoints committed under
`results/20260905-burgers-matched-semigroup-screen-e479174-3seed-cpu-r1/cells/s<seed>/checkpoints/`
for seeds `31415`, `271828`, and `161803`.  Rebuild each model with the
recorded configuration: `N=64`, `L=2*pi`, `nu=0.01`, `hidden=32`,
`ode_steps=12`, `radius=2`, and the label's own `query_conditioned` flag.

Regenerate the immutable screen cache from `--data-seed 20260902` and require
its digest to equal
`30c8443e37fa9b0d97e2b964699e8c0044f8f637232acf29abf915f19ac2de8d`.
Abort if it differs.  Evaluate on the first 16 validation initial conditions,
which is the same sample set the frozen runner used for its composition
endpoint, and additionally report all 50 validation initial conditions.

No checkpoint is selected, modified, or re-saved.  No cache is written other
than the regenerated screen cache, whose digest must match the value above.

## The three measured paths

Let $S^{[c]}_{\tau}$ denote one model call that advances the state by duration
$\tau$ while presenting $c$ to the control channel, and write $d=T/\tau$ for
the composition depth at horizon $T$.  All comparisons give both compared
paths the same number of right-hand-side evaluations.

1. **Frozen semantics.** Compare the composed path $(S^{[\tau]}_{\tau})^{d}$
   against the direct path $S^{[T]}_{T}$ integrated with $d\cdot\texttt{ode\_steps}$
   substeps.  This must reproduce the committed screen numbers.
2. **Matched conditioning.** Compare the same composed path against
   $S^{[\tau]}_{T}$, that is the direct path conditioned at the composed
   path's own lag rather than at the horizon.  Both paths then integrate one
   and the same vector field, so whatever remains is nonuniform Runge--Kutta
   truncation and not temporal structure.
3. **In-range cross-lag composition.** Compare $(S^{[0.04]}_{0.04})^{T/0.04}$
   against $(S^{[0.08]}_{0.08})^{T/0.08}$ with substeps chosen so both paths
   use $4\cdot(T/0.04)\cdot\texttt{ode\_steps}$ evaluations.  Both lags lie
   inside the evaluated range and each individual call is conditioned on a
   value the model saw during training, so no extrapolation is involved.  For
   one shared generator the two paths must agree to integrator accuracy.

Path 3 is the decisive measurement.  Path 1 asks whether asking for a whole
duration at once agrees with asking for it in pieces, which is the correct
semigroup question but is contaminated by extrapolation.  Path 3 asks whether
two *in-range* members of the family are iterates of one generator.

## Frozen interpretation rules

Define the per-seed, per-horizon **integrator floor** as the autonomous
model's path-2 defect, which contains no conditioning effect by construction
because the autonomous control channel is zeroed.  Declare a query-time defect
**collapsed** when it is within a factor of `10` of that floor and
**structural** when it exceeds the floor by at least a factor of `100`.

- If the query-time path-2 defect is collapsed while its path-1 defect remains
  structural, then the screen's headline defect magnitude is dominated by
  out-of-range conditioning, and the Burgers structural claim must be restated
  in terms of path 3 rather than path 1.
- If the query-time path-3 defect is structural, the query-time family fails
  the composition law strictly inside its trained lag range, and the mechanism
  claim survives without relying on extrapolation.
- If the query-time path-3 defect is collapsed, the Burgers query-time model
  has effectively learned to ignore its control channel in range.  The
  Burgers composition advantage would then be an extrapolation artifact and
  must be withdrawn, leaving only the Fisher lane as mechanism evidence.
- If the autonomous path-2 defect is not itself small relative to the
  autonomous path-1 defect, nonuniform Runge--Kutta truncation is not
  negligible at these horizons and every defect number in this lane needs a
  substep-refinement control before interpretation.

These rules fix the reading of the result before it is observed.  This lane is
exploratory, cannot be pooled with the locked Fisher decision, and cannot
change the frozen `0.90` accuracy threshold or the completed screen's primary
endpoint.
