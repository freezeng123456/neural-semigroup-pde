# Burgers direct-map control protocol

Status: frozen exploratory protocol, written before observing any result from
the direct time-conditioned map.

## Question

`CURRENT_METHOD_NOVELTY_PRIOR_ART.md` names the comparison this project has
never run.  Its factorial design is

| Model | Physics anchor | Autonomous continuous flow? |
|---|---:|---:|
| A | yes | yes |
| B | no / query-conditioned | yes |
| C | yes | **no** |

and it states plainly:

> The indispensable comparison is A versus C.  If A wins only against B, the
> result supports the double-well anchor, not semigroup novelty.

Every experiment in this repository to date is A versus B.  On Fisher--KPP and
on Burgers alike, B is still a continuous-time model integrated by RK4; only
its vector field reads the requested lag.  So *autonomy* and *continuous-time
integration* have never been separated, and no result yet distinguishes the
learned flow formulation from an ordinary direct operator that maps
$(u,\tau)\mapsto u(t+\tau)$.

This lane runs that control on Burgers at the adopted minimal architecture.

## Model C

C keeps the physics and the parameter budget and removes the flow:

$C_\theta(u,\tau)=e^{\tau\nu D_2}u+\tau\bigl(-D_0f_\theta(\operatorname{patch}(u),\tau)\bigr)$.

- `D_2` is the same three-point periodic Laplacian the flow models use, applied
  exactly through its discrete-Fourier eigenvalues
  $\lambda_k=(2\cos(2\pi k/N)-2)/\Delta x^2$.
- `D_0` is the same centered periodic divergence.
- $f_\theta$ is the **same flux network** as the adopted minimal generator:
  patch radius `0`, hidden width `8`, one hidden layer, plus the control
  channel.  C therefore has exactly `33` parameters, identical to A and B.
- Both terms preserve the spatial mean: $\lambda_0=0$, so the exponential fixes
  the constant mode, and a discrete divergence sums to zero.

C applies the flux **once** per call.  There is no ODE, no substepping, and no
fixed generator for the nonlinear part, so its maps at different requested
lags are not iterates of one flow.

One property is deliberate and must be read carefully: C's *linear* part
composes exactly by construction, because $e^{s\nu D_2}e^{t\nu D_2}=e^{(s+t)\nu D_2}$.
Any composition defect C exhibits is therefore attributable entirely to its
learned nonlinear increment, which is the cleanest available attribution.

## Frozen conditions

- **A and B are not retrained.**  They are the committed frozen checkpoints of
  `results/20260905-burgers-minimal-generator-ab-3seed-cpu-r1/`.
- C trains under exactly the screen's conditions: the immutable cache
  regenerated from `--data-seed 20260902` with digest
  `30c8443e37fa9b0d97e2b964699e8c0044f8f637232acf29abf915f19ac2de8d`, seeds
  `31415`, `271828`, `161803`, 1,000 training pairs, 50 validation
  trajectories, training lags `0.025`--`0.1`, unseen evaluation lags `0.04`
  and `0.08`, horizons `0.4` and `0.8`, 100 epochs, batch size 64, AdamW
  `1e-3` with `1e-5` weight decay, validation every ten epochs, and checkpoint
  selection by mean one-step validation MSE on the four training lags only.

**Work is not matched and no attempt is made to match it.**  A and B spend 48
flux evaluations per call, 12 RK4 substeps of four stages; C spends one flux
evaluation and one Fourier pair.  Equal-work matching is not defined for a
one-shot direct map.  The counts are reported with every number.  Note the
asymmetry runs against the flow formulation: C is the cheaper model, so a tie
on accuracy is a stronger statement than a tie at matched work would be.

## Endpoints

Primary: geometric-mean rollout MSE ratios `A/C` and `B/C` over the four
unseen-lag/horizon cells, pooled over three seeds, against the project's
`0.90` material threshold.

Structural, the reason this control exists:

1. in-range cross-lag consistency, comparing $C_{0.04}^{T/0.04}$ against
   $C_{0.08}^{T/0.08}$, to be read against A's value of exactly zero and B's
   `0.7456%` of the state norm;
2. direct-versus-composed consistency, comparing $C_T$ against
   $C_\tau^{T/\tau}$;
3. unseen-lag prediction spread, the largest relative spread of rollout MSE
   across the two unseen lags at a fixed horizon.

Also reported: spatial-mean drift, quadratic-energy change, the one-step
selection metric, and parameter count.

## Frozen decision rules

**Accuracy.** `autonomy_has_no_accuracy_advantage_over_direct_map` when the
pooled `A/C` ratio exceeds `0.90`, or meets it in fewer than two of three
seeds; `autonomy_has_material_accuracy_advantage_over_direct_map` otherwise.

**Structural specificity.** Using the band already frozen for the attribution
lane, with the integrator floor taken to be A's matched-conditioning defect
from the committed minimal lane: C's in-range cross-lag defect is `structural`
when it exceeds that floor by at least `100`, and `collapsed` when it is
within `10` of it.

- `structural_property_is_specific_to_autonomy` when C's in-range defect is
  `structural` in all three seeds.  Exact in-range composition and lag
  consistency are then properties of the autonomous flow that an ordinary
  direct time-conditioned map does not share, and the conditional structural
  claim the novelty audit allows is available.
- `structural_property_is_not_specific_to_autonomy` otherwise.  In that case
  the Burgers structural result distinguishes A from B but not from a direct
  operator, and the claim must be narrowed again rather than defended.

Both outcomes are results.  Neither may be resolved by trying a different C.

This lane is exploratory, cannot be pooled with the locked Fisher decision,
and does not alter the completed screen, the attribution lane, the ablation,
or the adopted reduction.  Its aggregator refuses inputs recorded at any other
architecture.
