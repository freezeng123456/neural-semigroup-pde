# Matched Burgers semigroup screen

Status: frozen exploratory protocol before observing the new split-generator
results.

## Purpose

This screen is the first transport--diffusion boundary test after Fisher.  It
does not reuse the legacy Burgers latent-gradient-flow/ResNet comparison.
Both new models use the periodic split generator

\[
F_\theta(u;q)=-D_0 f_\theta(\operatorname{patch}(u),q)+\nu D_2u,
\]

where `D_0` is a centered periodic divergence and `D_2` is the periodic
Laplacian.  Consequently, both models preserve the spatial mean up to floating
point error and share the exact viscous anchor.  The learned flux is the only
trainable spatial component.

Model A sets the final control channel to zero and integrates one autonomous
generator.  Model B has exactly the same parameters and feeds the requested
query lag through that channel.  The initial parameter tensors, optimizer,
training examples, RK4 work, and validation rule are matched.

## Frozen exploratory matrix

- seeds: `31415`, `271828`, and `161803`;
- periodic grid: `N=64`, `L=2*pi`, `nu=0.01`;
- shared data seed: `20260902`;
- 1,000 variable-lag training pairs and 50 validation trajectories;
- training lags: `0.025`, `0.05`, `0.075`, `0.1`;
- unseen evaluation lags: `0.04`, `0.08`;
- horizons: `0.4`, `0.8`;
- 100 epochs, batch size 64, AdamW, and 12 RK4 substeps per call;
- checkpoint selection uses mean one-step validation MSE on the four training
  lags only; the unseen evaluation lags and reported horizons are never used
  for selection;
- one GPU per seed; the two paired models run sequentially inside that seed's
  task so they never contend for one GPU.

Primary endpoint: geometric-mean rollout-MSE ratio `A/B`.  Structural
endpoints: equal-work composition defect, spatial-mean drift, and physical
quadratic-energy change.  This is exploratory and cannot be pooled with the
formal Fisher decision.

Before the three-seed matrix, one reduced single-GPU smoke must confirm CUDA
execution, exact parameter matching, finite forward/backward values, mean
preservation, checkpoint creation, and parseable summary output.
