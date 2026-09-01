# Fisher--KPP learned-trajectory generator screen

Status: completed frozen exploratory screen; results are reported in
`FISHER_GENERATOR_TUBE_RESULTS.md`.

## Question

The pointwise generator intervention reduced generator residual on its training
states but did not transfer to later rollout states.  This screen asks whether
sampling the generator loss on the current learned path reduces the
trajectory-localized residual and, secondarily, the locked-cache rollout
error.

For each training initial state, the current autonomous Model A produces
detached production-RK4 snapshots at

\[
t_j\in\{0,0.3,0.6,0.9,1.2\}.
\]

Let \(q_j\) be the normalized composite-trapezoid weights on this grid.  The
new loss is

\[
L_{\mathrm{tube}}
=\sum_jq_j\frac1N\left\|
F_{\theta,h}^{u}(y_{\theta,h}(t_j))-A_h(y_{\theta,h}(t_j))
\right\|_2^2,
\qquad
L=L_{\mathrm{step}}+0.01L_{\mathrm{tube}}.
\]

The snapshots are refreshed from the current model on every batch.  They are
detached before evaluating the residual, so gradients act on the vector field
at the sampled states but do not backpropagate through the rollout.  This is a
discrete on-policy collocation update, not full differentiation of the
self-dependent trajectory functional.

The same `0.01` weight is retained to isolate the effect of the state
distribution.  It will not be tuned or swept.

## Frozen comparison

- seeds: `31415`, `271828`, `161803`;
- model: autonomous `latent` only;
- control: the corresponding frozen unregularized Model A checkpoint;
- same per-seed training/validation cache, initialization seed, architecture,
  optimizer, 100 epochs, batch size 64, and variable training lags;
- no validation or locked-test state is used to construct the tube loss;
- checkpoint selection remains rollout validation at lag `0.1`;
- read-only evaluation uses the first 128 samples of the existing independent
  Fisher cache at lags `0.075`, `0.15` and horizons `1.2`, `2.4`, `4.8`;
- this is exploratory and does not change the formal Fisher decision.

## Readouts and fixed interpretation

The primary readout is the paired ratio of normalized-trapezoid weighted raw
generator RMS on each checkpoint's own learned path over `[0,1.2]`:

\[
R_{\mathrm{tube}}
=\frac{\widehat L_{\mathrm{tube}}(A+\mathrm{Tube})^{1/2}}
{\widehat L_{\mathrm{tube}}(A)^{1/2}}.
\]

Secondary readouts are the reference-trajectory generator ratio, each locked
lag/horizon rollout-MSE ratio, their geometric mean, and bound violations.

- three-seed geometric mean `R_tube <= 0.90` and rollout-MSE geometric mean
  below `1`: evidence that controlling the learned-path residual helps
  prediction;
- `R_tube <= 0.90` but rollout MSE not improved: the residual term is not
  sufficient, so stability is the next mechanism;
- `R_tube > 0.90`: this detached tube objective did not materially control the
  target term and will not be scaled up or weight-swept.

Completion requires one successful GPU smoke for the changed path sampler,
normal exit and parseable paired results for all three seeds, and unchanged
training and locked-test cache hashes.
