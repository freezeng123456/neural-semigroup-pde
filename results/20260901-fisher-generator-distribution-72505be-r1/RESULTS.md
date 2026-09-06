# Fisher generator distribution diagnostic results

This evaluation-only diagnostic compared the frozen baseline A checkpoint with
both the rollout-selected `best` and epoch-100 `final` A+Gen checkpoints.  No
model was trained, tuned, or selected in this diagnostic.

## Geometric-mean residual ratios across three seeds

| state distribution | best A+Gen / A | final A+Gen / A |
|---|---:|---:|
| training initial | 1.0094 | **0.9806** |
| training one-step target | 1.0184 | **0.9801** |
| validation initial | 1.0089 | **0.9789** |
| locked time 0.0 | 1.0099 | **0.9801** |
| locked time 0.6 | 1.0342 | 1.0440 |
| locked time 1.2 | 1.0588 | 1.0969 |
| locked time 2.4 | 1.0493 | 1.1431 |
| locked time 4.8 | 1.0693 | 1.0367 |

Raw and normalized residual ratios coincide here because both checkpoints
represent the same input states to numerical precision; the reference
generator denominator is therefore shared.  Normalization is not the cause of
the negative result.

The epoch-100 checkpoint shows that optimization did work on the distribution
used by the loss: residual is about 2% lower on training initial/target states,
validation initial states, and locked initial states.  The sign reverses after
the model has rolled forward for only `0.6` time units and is substantially
worse at intermediate/long times.  Rollout validation selected earlier epochs
(`25`, `30`, or `80`), which reduced the later deterioration but also removed
the training-state generator improvement.

Gradient tertiles do not explain the final-checkpoint failure: locked ratios
from low to high gradient were `1.0708`, `1.0531`, and `1.0491`.  The failure is
therefore primarily a trajectory-distribution problem rather than a monotone
high-gradient effect.

## Scientific conclusion

Pointwise generator agreement on the one-step data manifold is not sufficient
for long-horizon control.  The next theory/algorithm step should control the
generator mismatch along a tube containing the learned rollout, or derive a
stability-weighted trajectory bound that identifies which visited states need
supervision.  A loss-weight sweep is not justified by these results.

## Execution evidence

- best-checkpoint array `23620244_[0-2]`: all `COMPLETED`, exit `0:0`;
- final-checkpoint array `23620402_[0-2]`: all `COMPLETED`, exit `0:0`;
- remote lightweight result archive SHA-256:
  `451d5a16d2e0a0fb7657b727bf8838a40c068822e153f2739e76306efdebb7cb`.
