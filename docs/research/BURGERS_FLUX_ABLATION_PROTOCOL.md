# Burgers flux-generator ablation protocol

Status: frozen exploratory protocol, written before observing any ablation
result.

## Question

The matched Burgers screen fixed one architecture without justifying its
size.  The learned generator is

$F_\theta(u;q)=-D_0f_\theta(\operatorname{patch}(u),q)+\nu D_2u$,

where `D_0` is a centered periodic divergence and `D_2` the periodic
Laplacian.  Three of its choices are unargued:

1. a patch radius of `2`, giving the flux network a five-point stencil, even
   though the exact viscous Burgers flux $u^2/2$ is **pointwise**;
2. two hidden layers of width `32`, giving 1,313 parameters for a scalar
   one-dimensional flux;
3. the fixed viscous anchor, whose contribution has never been measured
   against a model that must learn diffusion through the flux divergence.

This ablation asks which of these components carries the result and which is
dead weight, so that any later theorem or paper claim describes the smallest
structure that actually works.

The query-time control channel is deliberately **not** an axis here.  The
screen and `BURGERS_QUERY_CONDITIONING_ATTRIBUTION_RESULTS.md` already
established that it does not improve accuracy — geometric-mean rollout-MSE
ratio `0.9933` in the autonomous model's favour — and that its only structural
effect is to break in-range composition by about `1.3%` of the state norm.
Every cell below therefore uses the autonomous generator.

## Frozen grid

Seven variants, three seeds each, all autonomous.

| Label | Patch radius | Hidden width | Hidden layers | Viscous anchor | Parameters |
|---|---:|---:|---:|---|---:|
| `r2_h32_l2_anchor` | `2` | `32` | `2` | yes | `1313` |
| `r1_h32_l2_anchor` | `1` | `32` | `2` | yes | `1249` |
| `r0_h32_l2_anchor` | `0` | `32` | `2` | yes | `1185` |
| `r0_h8_l2_anchor` | `0` | `8` | `2` | yes | `105` |
| `r0_h8_l1_anchor` | `0` | `8` | `1` | yes | `33` |
| `r2_h32_l2_noanchor` | `2` | `32` | `2` | no | `1313` |
| `r0_h32_l2_noanchor` | `0` | `32` | `2` | no | `1185` |

`r2_h32_l2_anchor` is the screen architecture and serves as the reference.
Removing the anchor sets the model's viscosity to zero while the reference data
keep `nu=0.01`, so those two variants must recover diffusion through the flux
divergence.  A pointwise flux cannot represent a second derivative at all, so
`r0_h32_l2_noanchor` is a deliberate negative control.

This is an ablation, not a parameter-matched comparison.  Parameter counts are
reported for every cell and no cell is padded to match another.

## Fixed conditions

Everything outside the architecture is held at the screen's frozen values:
the immutable cache regenerated from `--data-seed 20260902` with digest
`30c8443e37fa9b0d97e2b964699e8c0044f8f637232acf29abf915f19ac2de8d`, seeds
`31415`, `271828` and `161803`, 1,000 training pairs and 50 validation
trajectories, training lags `0.025`--`0.1`, unseen evaluation lags `0.04` and
`0.08`, horizons `0.4` and `0.8`, 100 epochs, batch size 64, AdamW at `1e-3`
with `1e-5` weight decay, validation every ten epochs, and 12 RK4 substeps per
call.  Checkpoint selection uses mean one-step validation MSE on the four
training lags only; the unseen lags and reported horizons never enter
selection.

## Endpoints

Primary: geometric-mean rollout MSE over the four unseen-lag/horizon cells,
reported as a ratio to `r2_h32_l2_anchor` at the same seed.

Structural, reported per cell and never traded against the primary endpoint:
spatial-mean drift, equal-work nonuniform composition defect under matched
conditioning, and quadratic-energy change.  Parameter count and training time
are recorded.

## Frozen decision rule

Using the project's existing 10% materiality convention from
`AUTO_RESEARCH_PROTOCOL.md`, classify each variant by its pooled
three-seed geometric-mean ratio $R$ to the reference:

- `equivalent` when $1/1.1\le R\le 1.1$;
- `degraded` when $R>1.1$;
- `improved` when $R<1/1.1$.

A variant additionally fails on structure, whatever its accuracy, if its
spatial-mean drift exceeds `1e-6` or its composition defect exceeds ten times
the reference defect at the same seed and cell.

**A component counts as removable only if the variant that removes it is
`equivalent` or `improved` and passes the structural check in all three
seeds.**  The recommended minimal generator is then the lowest-parameter
variant satisfying that condition.  If no reduced variant qualifies, the
screen architecture is kept and this lane reports that its size is justified.

The rule is fixed here before any ablation cell runs.  This lane is
exploratory, cannot be pooled with the locked Fisher decision, and does not
alter the completed screen or its `0.9933` primary endpoint.  The screen's
runner is not edited, so the committed screen results stay reproducible from
their recorded source digests.
