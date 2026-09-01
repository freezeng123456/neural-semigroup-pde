# Fisher learned-tube attribution protocol

Status: frozen exploratory checkpoint-only protocol before observing these
common-path results.

## Question

The learned-tube intervention reduced its own-path generator residual by
about `5.8%` and the locked-cache rollout MSE by about `17.6%`.  The registered
generator threshold was not met, so this follow-up does not tune the loss or
train another model.  It asks whether the secondary MSE improvement is
associated with a common-state vector-field correction, a change of trajectory
distribution, or a change in local amplification.

## Frozen inputs and matrix

Use the three completed seeds `31415`, `271828`, and `161803`.  For each seed,
reuse the frozen baseline A checkpoint, the completed A+Tube checkpoint, and
the immutable formal Fisher test cache.  Evaluate the first 128 test
trajectories at times `0`, `0.3`, `0.6`, `0.9`, and `1.2`.

Both vector fields are evaluated on exactly the same three state tubes:

1. the reference PDE trajectory;
2. the baseline A learned trajectory;
3. the A+Tube learned trajectory.

At each time and state source, report the physical generator residual, the
`sinusoidal_0.01` empirical one-sided quotient, and the component of the
generator defect that injects the current trajectory error.  Aggregate in
time with the normalized composite-trapezoid weights.

## Interpretation

- a Tube/baseline residual ratio below one on the reference and both learned
  paths supports a common-state vector-field correction;
- improvement only on the Tube path identifies trajectory redistribution;
- a lower harmful defect component or one-sided quotient with a modest total
  residual change supports stability/directional amplification as the missing
  explanation;
- no common-path or stability improvement means the MSE result must be treated
  as checkpoint-selection or optimization evidence rather than generator
  attribution.

This experiment is exploratory, performs no training or checkpoint selection,
and cannot change the formal Fisher decision or the learned-tube `0.90`
generator threshold.
