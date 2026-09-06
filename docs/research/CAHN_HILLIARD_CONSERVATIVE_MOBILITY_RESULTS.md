# Cahn--Hilliard conservative mobility: completed exploratory results

Date: 2026-09-07

The frozen lane in `CAHN_HILLIARD_CONSERVATIVE_MOBILITY_PROTOCOL.md`
is complete.  Protocol commit `1354ce9` predates training.
All values are `exploratory=true` and `do_not_use_for_formal=true`.

Canonical root:
`/jizhicfs/yuyechen/neural_semigroup/20260907-cahn-hilliard-conservative-mobility-h20-r1`.

## Outcome

Explicit RK4 with 30 substeps **does not train**.  Epoch 1 loss was
`5.17e-2`.  By epoch 5 the step loss was `NaN`.  No
`a_conservative_best.pt` was written.  The trained-model mass rule
therefore has no checkpoint to evaluate.

The same architecture at the frozen initialization is finite:

| Readout | Untrained field | Frozen reference |
|---|---:|---|
| Pooled rollout MSE | `0.236` | recorded, no 0.90 gate |
| Max mass drift | `5.66e-7` | `1e-6` |
| Seeds | `42/137/2718` identical | zero output layers |

Registered decision: training diverged.  The untrained field meets
`conservative_mobility_conserves_mass` as an architecture diagnostic,
not as a learned generator.

## What this means

`-D^\top M D` conserves mass when the integrator stays finite.  An
explicit RK4 budget copied from Allen--Cahn is not enough to train a
fourth-order conservative field on the locked lags.  That is a
time-discretization failure, not a reason to revert to diagonal `K`.

The untrained MSE `0.236` is better than the exploded large-step
`C_phys` and worse than the small-step split (`1.5e-4` at `T=0.6`).
A later generator has to beat the ETD-RK4 cache, and it has to be
trained with a stable treatment of the biharmonic part.

## Claim boundary

Supported:

- the conservative stencil has zero-mean vector fields and sub-`1e-6`
  mass drift while finite;
- the frozen RK4-30 training loop diverges on this PDE.

Not supported:

- any learned Cahn--Hilliard accuracy number;
- any claim that conservative mobility is sufficient without a stiff
  integrator;
- quietly lowering the lag, width, or step count to make the loss
  finite.

## Next designed experiment

Keep \(-D^\top M D\).  Integrate the linear biharmonic semigroup
exactly and apply one conservative nonlinear increment, the analogue
of the reaction--diffusion `C_{\mathrm{phys}}` that actually ran.
Do not restart RK4-30 with a smaller learning rate.
