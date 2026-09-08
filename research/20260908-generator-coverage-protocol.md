# Generator identification: H20 exploratory protocol (frozen before full runs)

Continue the shared known `-u^3` experiment. Both A (autonomous) and B (requested-lag-conditioned) use the same 65-parameter network and the same known diffusion/cubic term. A has 49 effective parameters because its 16 lag input weights are inactive. No unknown reaction labels are supplied to ordinary training.

## Data and primary comparisons

Use fresh validation/test/stress seeds 2026090801/02/03. Training seeds are 31415, 271828, 161803 with disjoint data seed offsets from the September 7 screen. Reference FP64 RK4 `dt=.002` is checked against `.001`. N=64, domain `[0,2pi)`, diffusion `.02`, unknown reference source remains `u-u^3+.2sin(3u)`.

Four main modes: initial one-step; teacher-forced reference intermediate one-step; learned rollout with temporal gradient detached before each step; fully differentiated short rollout. Length L=4. Lags `.05,.1,.2` are assigned per trajectory. All use 128 supervised field predictions and `128*64*4` scalar reaction evaluations per Adam update, 400 updates, learning rate `.01`. Initial uses 128 independent initial states/128 unique pairs; trajectory modes use a nested 32 initial states with four consecutive pairs each (128 unique pairs). Thus unique initial-state diversity differs explicitly. An `initial_repeat` control uses the same nested 32 initial states repeated four times (32 unique pairs). This control separates data diversity from reuse but does not make every causal contrast unconfounded. Teacher vs detached vs unroll uses exactly the same 32 trajectories and target snapshots.

Record actual synchronized optimizer-step seconds separately from validation/evaluation time. Full unroll necessarily changes backward graph cost; equal forward calls are not a claim of equal wall time. All methods receive the same update budget; any claim about speed must use measured time.

Wave 1: Euler heat/reaction splitting, four reaction evaluations per map, four main modes plus initial_repeat, 3 seeds, A/B = 30 full cells. Wave 2: midpoint reaction with two split substeps (four reaction evaluations per map), four main modes, 3 seeds, A/B = 24 full cells. An explicitly privileged same-network source regression diagnostic uses true scalar source labels on a uniform grid, 3 seeds, A/B = 6 cells; it is not part of the unknown-identification comparison. Further attempts must be named and frozen separately before running, with a bounded scope and new holdout seeds if selected adaptively.

## Model selection and evidence

Validation every 10 updates plus the first/final update. Save final, epoch120, best one-step validation, and best short-rollout validation checkpoints. Primary checkpoint is best rollout validation, averaged over independent trajectories at horizons `.4,.8` using lags `.1,.2`. This choice is fixed before test inspection. Tests are never used for checkpoint selection. Report other checkpoints as optimization/selection diagnostics.

Primary test is geometric mean of endpoint MSE at lags `.075,.15`, horizons `1.2,2.4`; secondary horizon `.6` and lag `.3` are stress/extrapolation diagnostics. Report high-amplitude independent stress, observed invariant-bound violations and energy monotonicity. These observations are not mathematical certificates. Material A/B criterion within a named mode: pooled primary ratio <=.90, at least 2/3 seeds <=.90, no seed >1.05, no bound degradation >.02 and energy monotonicity degradation >.02. The many-mode search is exploratory; a passing cell/mode is not formal confirmation or a novel theorem.

Reaction accuracy on a common uniform grid and reference/deployed state distributions, state histograms, refinement and unequal-partition composition defects diagnose mechanism. In B, tau=0 reaction is a diagnostic extrapolation and does not define a proven generator. Midpoint and Euler have different heat-call costs despite equal reaction calls; report measured times. Structural curves are indexed by split substeps; midpoint has two reaction calls per substep. No structural-only improvement can pass the accuracy criterion.

## Completion

Single H20, single training process. Preserve source commit/hash, cache hash, finite changed weights, all checkpoint files, metrics, completion/exit markers, full root manifests. Verify all recovered files and publish full results separately from code. Do not claim completion merely from launch or GPU usage.
