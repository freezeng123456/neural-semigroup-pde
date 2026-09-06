# Wave 2 direct-map control at matched work

Status: frozen exploratory protocol, written before observing any result
from a direct time-conditioned map on the Wave 2 reaction--diffusion PDE.

## Question

Wave 2 is the only lane in this repository that still carries a large
accuracy number.  On the hard-boundary cells, autonomous long-rollout MSE
was 8--11 times lower than query-time, with ratios `0.125`, `0.110`, and
`0.089` across Neumann, Dirichlet, and Robin.  That comparison is A versus
B: both models are continuous-time RK4 flows.  The novelty audit's
indispensable comparison is A versus C, a direct time-conditioned map, and
it has been run only on Burgers.

On Burgers the unmatched A/C ratio `0.699` retracted at matched work to
`1.008`.  Wave 2's accuracy figure has never faced that control, on the
PDE family where autonomy actually beat query-time.  This lane runs that
control, and it matches work from the first cell so the Burgers confound
cannot recur.

It also carries the corrected refinability measurement that
`BURGERS_WORK_MATCHED_CONTROL_RESULTS.md` could not adopt: the earlier
rule demanded that the query-conditioned flow refine away a defect that is
irreducible by construction, and it asked a first-order Euler integrator
for a 100-fold separation on a sweep that stopped at 16 substeps.  The
rules below apply refinement only to the autonomous model, extend the
sweep to 128, and set the separation factor from the integrator order.

## PDE and architecture

The reference equation, grid, caches, seeds, and stencil MLP are Wave 2's:

\[
u_t=0.02\,u_{xx}+u-u^3,\qquad x\in(0,1),\qquad N=33.
\]

Hard enforcement only.  All three boundary families.  The neural net is
the Wave 2 three-point stencil MLP, hidden width 32, 1,249 parameters.
Penalty and unconstrained cells are not rerun; they are not the question.

## Models

All five trained models in a seed and family start from one parameter
fingerprint and then train independently.

| Label | Temporal object | Deployed map | Net evaluations per call |
|---|---|---|---:|
| `A` | autonomous field, RK4, 8 steps | Wave 2 production flow | 32 |
| `B` | query-conditioned field, RK4, 8 steps | Wave 2 production flow | 32 |
| `A'` | autonomous field, explicit Euler, 1 step | \(u+\tau F_\theta(u)\) | 1 |
| `B'` | query-conditioned field, explicit Euler, 1 step | \(u+\tau F_\theta(u;\tau)\) | 1 |
| `C` | direct residual map | \(C(u,\tau)=P_B\bigl(u+N_\theta(u,\tau)\bigr)\) | 1 |

\(P_B\) is Wave 2's parameter-free affine boundary reconstruction.  \(C\)
uses the same stencil MLP as the vector field, including the duration
feature \(\tau/0.08\), but the output is a state residual, not a
derivative, and it is not multiplied by \(\tau\).  Maps at different
requested lags are therefore not iterates of one flow.  \(C\) is not
\(B'\): \(B'\) scales a vector-field output by \(\tau\); \(C\) adds an
unscaled residual whose magnitude the network must learn from the
duration feature.

`A` and `B` exist only as a descriptive unmatched-work diagnostic and as
a check that this reimplementation still sees Wave 2's A-versus-B
direction.  No accuracy verdict may be read from `A/C` or `B/C`.

## Frozen conditions

- data seed `515151`, Wave 2 `data_identity`, one immutable cache per family;
- seeds `31415`, `271828`, `161803`;
- 512 / 64 / 128 train / val / test samples;
- training durations `0.02, 0.04, 0.08, 0.16`;
- 150 epochs, batch 64, Adam `1e-3`, weight decay `1e-6`;
- validation every 5 epochs and on the final epoch;
- checkpoint selection by mean one-step validation MSE on the four
  training durations only;
- hard-boundary threshold `5e-6`.

No width, depth, lag, horizon, penalty, or learning-rate search.

## Endpoints

Primary accuracy, at exactly one net evaluation per call:

geometric-mean long-rollout MSE ratios `A'/C` and `B'/C` over Wave 2's
four unseen pairs \((\tau,H)\in\{0.03,0.06\}\times\{0.24,0.48\}\),
pooled over three seeds, against the project's `0.90` material
threshold.  Families are not pooled with each other.

Structural, the reason the control exists:

1. in-range cross-lag consistency at horizon `0.16`, comparing four
   steps of `0.04` against two steps of `0.08`;
2. a refinement sweep of that defect at Euler substeps
   `1, 2, 4, 8, 16, 32, 64, 128` for `A'` and `B'`;
3. equal-substep composition of `A'`, which must be exactly zero;
4. hard-boundary residuals on Wave 2's endpoint-stress states.

`C` has no refinement axis.  Its cross-lag defect is a single number.

Unmatched `A/C` and the Wave 2-style `A/B` long-rollout ratios are
reported as descriptive only.

## Frozen decision rules

**Accuracy, matched work.**  For each family separately:
`flow_has_material_accuracy_advantage_over_direct_map` when the pooled
`A'/C` ratio is at most `0.90` and at least two of three seeds meet
`0.90`; `flow_has_no_accuracy_advantage_over_direct_map` otherwise.
Unmatched `A/C` cannot change this verdict.

**Refinability, corrected.**  Apply the refinement requirement to the
autonomous model only.

- `autonomous_defect_is_refinable` when, in every family, the geometric
  mean of `A'`'s in-range cross-lag defect at 1 substep divided by the
  value at 128 substeps is at least `64` (half the first-order ideal of
  `128`), and `A'`'s equal-substep defect is `0` in every seed.
- `query_time_has_irreducible_defect` when `B'`'s corresponding
  refinement gain is strictly less than `8` in every family.  That is
  the expected structural remainder: refining `B'` converges two
  different fields, one conditioned on `0.04` and one on `0.08`.
- `structural_ordering_is_specific_to_autonomy` when `C`'s deployed
  in-range defect exceeds `A'` at 128 substeps by at least `16` in
  every family.  Sixteen is an order-of-magnitude floor compatible with
  a first-order sweep; it is not the misspecified factor of `100` at
  16 substeps.

A negative accuracy verdict and a positive structural verdict together
are the Burgers conclusion on a second PDE.  The opposite pair would be
the first accuracy claim in the project that has survived a
work-matched direct map.  Both outcomes are results.  Neither may be
resolved by changing `C` or the integrator.

This lane is exploratory, cannot be pooled with the locked Fisher
decision, and does not alter Wave 2, the Burgers stack, or the certified
constants.  Its aggregator refuses summaries recorded at another
architecture, at an unmatched Euler budget, or without the 128-substep
sweep.
