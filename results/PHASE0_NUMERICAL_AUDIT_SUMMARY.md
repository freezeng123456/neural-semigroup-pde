# Phase-0 numerical audit summary

Status: **PASS**.  Phase 0 is diagnostic only; no checkpoint was retrained.

## Reproducibility

- Audit implementation commit: `b0e14cec276a52e641dd657fb2e347849bd8acc7`
- Unit/regression tests on the T4 environment: `29 passed`
- Reference runs:
  - `20260825-phase0-reference-label-floor-t02-b0e14ce`
  - `20260825-phase0-reference-label-floor-t20-b0e14ce`
- Three-checkpoint run:
  - `20260825-phase0-checkpoint-audit-3seed-b0e14ce`
- Checkpoint seeds: `42`, `123`, `2026`
- Checkpoint architecture: Fisher--KPP latent model with coercive
  `beta_V_floor=0.1`
- Hardware: Tesla T4; float64 reference solves run on CPU and checkpoint
  diagnostics run on CUDA.

Every run directory includes the command/launcher, commit, input checkpoint
hashes where applicable, environment/hardware identity, logs, status, and raw
JSON/CSV outputs.

## 1. Production-label error

The production label (`N=64`, `dt=0.005`, float32) was compared directly with
a fine reference (`N=512`, `dt=0.0005`, float64) for 20 deterministic smooth
initial conditions.

| Horizon | Comparison | Mean relative L2 | Maximum relative L2 |
|---:|---|---:|---:|
| 0.2 | production label vs fine reference | 6.2534e-5 | 2.7648e-4 |
| 0.2 | float32 vs float64 at production grid/time step | 4.1291e-7 | 7.8110e-7 |
| 0.2 | `dt=0.005` vs `dt=0.0005` at `N=512`, float64 | 1.9236e-11 | 5.0668e-11 |
| 2.0 | production label vs fine reference | 3.9163e-5 | 1.4820e-4 |
| 2.0 | float32 vs float64 at production grid/time step | 1.3223e-6 | 3.0051e-6 |
| 2.0 | `dt=0.005` vs `dt=0.0005` at `N=512`, float64 | 2.8819e-11 | 6.0624e-11 |

Conclusion: the production reference is numerically usable, but its dominant
uncertainty is the 64-point spatial discretization.  Neither `dt=0.005` nor
float32 is the main source.  The worst smooth-IC spatial errors are large
enough that the later grid-refinement/OOD phase remains mandatory.

## 2. Three-seed checkpoint audit at the production 30 RK4 steps

Values are mean ± sample standard deviation over the three independently
trained checkpoints.  Each checkpoint was evaluated on all 50 validation
trajectories for 20 model steps.

| Metric | Mean ± sample SD |
|---|---:|
| Rollout MSE | 1.2707595e-4 ± 8.2638466e-6 |
| Production fixed-steps semigroup defect, absolute L2 | 2.9474893e-7 ± 8.1187013e-10 |
| Production fixed-steps semigroup defect, relative L2 | 6.5984036e-8 ± 1.7763257e-10 |
| Equal-work semigroup defect, absolute L2 | 2.0171685e-7 ± 1.6711070e-9 |
| Equal-work semigroup defect, relative L2 | 4.5112916e-8 ± 3.7708957e-10 |
| Maximum learned-energy positive increment | 0 for all seeds |
| Maximum physical-energy positive increment | 0 for all seeds |

The equal-work diagnostic gives the direct path `(i+j)*30` RK4 substeps and
the composed path `i*30 + j*30` substeps.  Its small value shows that the
observed composition consistency is not created solely by giving the composed
path more RK4 work.  It remains a numerical approximate-consistency result,
not an exact-equality claim for RK4.

## 3. No-retraining RK4 sweep

Each row averages the three checkpoints.  `Output rel. difference` compares a
single model call with the 120-substep output.  Defects use one validation
state per checkpoint and 20 composition pairs.

| RK4 steps | Output rel. difference vs 120 | Production defect, abs L2 | Equal-work defect, abs L2 |
|---:|---:|---:|---:|
| 5 | 1.0680e-7 | 1.3780e-5 | 2.0871e-7 |
| 10 | 1.1184e-7 | 7.1472e-7 | 1.8715e-7 |
| 15 | 1.1382e-7 | 2.9788e-7 | 2.0865e-7 |
| 30 | 1.1274e-7 | 3.2198e-7 | 1.9341e-7 |
| 60 | 1.2501e-7 | 3.9463e-7 | 2.1610e-7 |
| 120 | 0 | 4.8874e-7 | 1.9921e-7 |

Interpretation:

1. Five fixed substeps are insufficient for long-duration direct calls under
   the production metric; the defect is about two orders of magnitude larger.
2. From 10--15 substeps onward, one-call outputs and equal-work composition
   errors are at a float32-scale plateau rather than showing meaningful
   improvement with more substeps.
3. The production choice of 30 substeps is conservative and is not the source
   of the approximately `1e-4` rollout MSE.
4. Increasing substeps beyond 30 raises cost without measurable prediction
   benefit in this checkpoint audit.

## 4. Cost

Per seed on the T4:

- Full 50-trajectory evaluation including equal-work diagnostics:
  `28.032 ± 0.305 s`.
- Six-point RK4 sweep with both defect definitions:
  `208.150 ± 2.242 s`.

The sweep is expensive because RK4 stages remain sequential in integration
time even though states and validation trajectories are batched in parallel.

## 5. Decision for the next phase

Phase 0 passes.  Proceed with parameter-matched ResNet/FNO baselines and
variable-time experiments.  Preserve these qualifications:

- learned energy and physical energy are distinct quantities even though both
  were monotone on this Fisher--KPP validation set;
- reference-label and rollout metrics currently use different normalizations,
  so a later unified relative-L2 metric is needed before direct error-floor
  claims;
- three seeds establish basic reproducibility but not a high-powered
  significance result;
- smooth low-frequency initial conditions do not replace the planned grid and
  OOD tests.
