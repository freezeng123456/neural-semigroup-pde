# Fisher--KPP three-seed baseline and inference optimization

This branch preserves the complete canonical outputs for the first formal
architecture-only Fisher--KPP experiment and the subsequent inference
optimization benchmark.

## Formal run

- Run: `20260824-fisher-coercive-3seed-fcd9e39`
- Training code: `fcd9e394ee87f229de57916271dcf42f0175bda4`
- Seeds: 42, 123, 2026; shared data seed: 42
- Device: one Tesla T4; seeds were executed sequentially
- Protocol: 100 epochs, deterministic, architecture-only losses, positive
  coercive floor `beta_V_floor=0.1`

Across the three paired seeds:

| Metric | Latent semigroup model | ResNet baseline |
|---|---:|---:|
| Rollout MSE, mean over seeds | 1.270759e-4 | 3.095887e-4 |
| Rollout MSE, sample SD over seeds | 8.263853e-6 | 6.692357e-5 |
| Mean paired MSE reduction | 57.62% | -- |
| Bound violation, mean | 0 | 8.580408e-5 |
| Numerical semigroup defect, absolute L2 mean | 2.948074e-7 | 1.621189e-1 |
| Geometric-mean defect ratio (baseline / latent) | 5.499151e5 | -- |
| Physical-energy monotonicity | 100% | 100% |

The learned-energy metric describes `model.energy`; it is not assumed to equal
the physical Fisher--KPP energy.

## Inference optimization benchmark

- Run: `20260824-inference-optimization-benchmark-1fd45f0`
- Before: `fcd9e394ee87f229de57916271dcf42f0175bda4`
- After: `1fd45f0b732852a7b8d720a3898fed6146815621`
- Device: Tesla T4
- Evaluation: 50 samples, 20 rollout steps

| Measurement | Before | After | Speedup |
|---|---:|---:|---:|
| Validation | 285.7005 s | 5.1336 s | 55.65x |
| Formal evaluation | 1036.5755 s | 21.3699 s | 48.51x |

The aggregate checker marked one near-zero diagnostic outside its relative
tolerance: the absolute-L2 semigroup defect changed by approximately
`2.1e-10`. Rollout, bound, energy, latent-state, and other semigroup metrics
were within tolerance. Phase-0 follow-up work will use an absolute tolerance
floor for diagnostics at the FP32 noise scale.

All per-seed checkpoints, histories, configurations, metrics, logs, provenance,
and completion markers are retained under their canonical run directories.
