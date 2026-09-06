# Allen--Cahn work-matched direct-map protocol

Status: frozen exploratory protocol, written before any cell of this
lane ran.  It executes the A-versus-C comparison that
`CURRENT_METHOD_NOVELTY_PRIOR_ART.md` and
`SEMIGROUP_CROSS_PDE_VALIDATION.md` call indispensable, now on the
home PDE, and it equalises work up front so the Burgers confound
cannot recur.

All values from this lane are `exploratory=true` and
`do_not_use_for_formal=true`.  The locked Allen--Cahn A/B attribution
and the locked Fisher decision are not reopened.

## Why this lane, and why not a search

The project's only material accuracy win is Allen--Cahn physics-anchor
versus no-anchor.  Both of those models are autonomous flows, so that
win does not isolate semigroup structure.  Burgers later ran the
missing A-versus-C comparison and then retracted the accuracy figure
once flux evaluations were matched.  Allen--Cahn has never had either
half of that pair.

This lane does not sweep widths, losses, lags, or epochs.  The
architecture, data, optimiser, and seeds are the attribution
protocol's frozen Allen--Cahn settings.  The only designed
interventions are the temporal law and the integrator budget.

## Models

Every model keeps the bounded decoder \(u= -1 + 2\sigma(z)\), the
fixed double-well \((1-u^2)^2/4\), the learned residual potential, the
periodic decoded-state interaction, and positive local mobility.  They
differ only in how the requested duration \(\tau\) is allowed to
change the map.

| Label | Temporal law | Integrator | Learned evaluations per call |
|---|---|---|---:|
| \(A\) | autonomous field \(f(z)\) | RK4, 30 substeps | 120 |
| \(A'\) | the same autonomous field | explicit Euler, 1 substep | 1 |
| \(B'\) | query-conditioned field \(f(z;\tau)\) | explicit Euler, 1 substep | 1 |
| \(C\) | exact heat semigroup plus one query-conditioned latent increment | one-shot | 1 |

Explicitly,

\[
A'(u,\tau)=\operatorname{decode}\bigl(z+\tau f(z)\bigr),
\qquad
B'(u,\tau)=\operatorname{decode}\bigl(z+\tau f(z;\tau)\bigr),
\]

\[
C(u,\tau)=\operatorname{decode}\bigl(z_{\mathrm{lin}}+\tau f(z_{\mathrm{lin}};\tau)\bigr),
\quad
u_{\mathrm{lin}}=e^{\tau\varepsilon^2\Delta}u,
\quad
z_{\mathrm{lin}}=\operatorname{encode}(u_{\mathrm{lin}}).
\]

\(C\) is the Allen--Cahn analogue of the Burgers direct map: the linear
diffusion piece is the exact periodic heat semigroup and therefore
composes, while the nonlinear increment is a single learned evaluation
that reads \(\tau\).  Iterating that increment at \(\tau/K\) would
turn \(C\) into a query-conditioned flow, so \(C\) is not refined at
evaluation time.  The only parameter-matched way to equalise work is
to bring the flows down to one Euler step.

\(B'\) and \(C\) receive the untrained \(A'\) mobility weights with a
zero \(\tau\) column, then train independently.  The extra 64
parameters are reported, not hidden.

## Frozen design

- **PDE:** 1D periodic Allen--Cahn, \(N=64\), \(L=2\pi\),
  \(\varepsilon=0.1\), reference step \(0.005\).
- **Data:** fresh variable-lag cache per seed; 1,000 training and 50
  validation trajectories; training lags \(0.025,0.05,0.10,0.20\);
  unseen evaluation lags \(0.075,0.15\); horizons \(1.2,2.4,4.8\).
- **Locked test:** 500 trajectories from `data_seed + 1000003`, never
  used for selection.
- **Seeds:** `42`, `137`, `2718`, paired with the same data seed.
  These are the attribution-protocol seeds so \(A\) can be compared
  with the historical A/B signature on a newly generated cache, not so
  that old checkpoints may be reused.
- **Training:** 100 epochs, batch 64, Adam \(10^{-3}\), weight decay
  \(10^{-5}\), validation every five epochs, architecture-only losses,
  `beta_v_floor=0`.
- **Selection:** validation-only at the attribution protocol's fixed lag
  `0.1`.  Locked tests start only after the four checkpoints of a seed
  exist.
- **Work:** \(A\) is the unmatched-compute reference.  Primary
  accuracy and structural rules use \(A'\), \(B'\), and \(C\) at one
  learned evaluation per call.

## Endpoints

Primary accuracy, at one learned evaluation per call: geometric-mean
rollout-MSE ratios \(A'/C\) and \(B'/C\) over the six
unseen-lag/horizon cells, pooled over three seeds, against the
project's \(0.90\) material threshold.

Also reported, and not eligible to rescue a failed accuracy rule:

1. unmatched \(A/C\) at the historical 30-step RK4 budget;
2. in-range cross-lag composition defect relative to state RMS, for
   every model, at the deployed budget;
3. an evaluation-time refinement sweep \(1,2,4,8,16\) Euler substeps
   for \(A'\) and \(B'\) only, against \(C\)'s single irreducible
   value;
4. the equal-substep in-range defect for \(A'\), which must be
   numerically zero because its field is autonomous;
5. physical Allen--Cahn free-energy monotone fraction;
6. bound violations, one-step selection MSE, parameter count, and
   wall clock.

## Frozen decision rules

**Accuracy at matched work.**
`flow_has_material_accuracy_advantage_over_direct_map` when the
pooled \(A'/C\) ratio is at most \(0.90\) and at least two of three
seeds agree.  Otherwise
`flow_accuracy_advantage_does_not_appear_at_matched_work`.

**Unmatched compute is diagnostic only.**  A pooled \(A/C\) ratio
below \(0.90\) is recorded as a compute effect if the matched rule
failed.  It is never adopted as a semigroup-accuracy claim.

**Structure.**
`autonomy_orders_composition_defect` when, in every seed, the
in-range cross-lag defect of \(A'\) is exactly zero at equal substeps
and strictly below both \(B'\) and \(C\) at the deployed budget.

**Refinability.**
`flow_defect_refines_direct_map_defect_does_not` when, in every seed,
the \(A'\) deployed-budget defect decreases monotonically over
`1,2,4,8,16` substeps and \(C\) has no refinement axis.

## Claim boundary

A positive matched-work accuracy rule would be the first evidence that
autonomy, not the double-well and not extra RK4 work, improves
Allen--Cahn prediction.  A negative matched-work accuracy rule, even
with a clean structural ordering, would close the semigroup-accuracy
chapter on the home PDE the same way Burgers closed it on a transport
problem.  The paper would then be a physics-anchor and
structure-preservation paper, not a prediction-advantage paper.
