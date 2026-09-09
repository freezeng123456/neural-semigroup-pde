# New equations: frozen exploratory protocol (2026-09-10)

This protocol is frozen before looking at trained-model results on either new PDE. The earlier scalar reaction-diffusion confirmation is separate and already observed. No optimizer sweep or test-driven equation/seed removal is permitted.

## Equations and hypotheses

Both problems use the periodic interval [0,2pi), N=32, DX=2pi/N. Conclusions are fixed-grid unless spatial convergence is supplied separately.

1. Nonlinear diffusion: u_t = kappa div(a(u) grad u), a(u)=.05+.8u^2. The reference uses a at the midpoint of each adjacent pair, a conservative edge flux and its divergence. Hypothesis: a duration-independent learned positive diffusivity pools inconsistent/noisy snapshot information and transfers to changed known kappa without retraining. Mass, quadratic energy and the maximum principle are separate structural endpoints.
2. Damped sine-Gordon: u_t=v, v_t=c^2 D2u-.15v-sin(u). Hypothesis: a duration-independent learned restoring force yields more consistent oscillatory long-time propagation and known-wave-speed transfer. Both models get (u,v), c and damping; no position-only disadvantaged baseline is used. Physical sine-Gordon energy is diagnostic, not structurally guaranteed by the learned force.

The learned law is a one-hidden-layer Tanh network (2 inputs, width 32, scalar output). Inputs are (local state, requested duration/.12); the second input is zero for autonomous. Effective parameters are 97/129 (autonomous/query); nominal parameters both 129. This is not an effective-capacity-matched attribution test. For diffusion a_theta=.02+1.5 sigmoid(net(midpoint,q)); for waves f_theta=2 tanh(net(u,q)). The exact nonlinear laws are used only by reference generation and post-training diagnostics, not passed to training as labels. These bounded-law priors are explicit and shared by both models.

## Numerical implementation

Diffusion uses SSPRK2 with h <= min(.01,.9 DX^2/(2 kappa*1.52)). This is a conservative stability bound using the architecture's global diffusivity bound, not a test-selected setting. Wave evolution uses Strang splitting between exact Fourier flow of the known linear damped wave system and a learned nonlinear velocity kick, max substep .03. The two models share all substeps and calls within each equation. Their computational work is not compared across different equations.

FP64 RK4 reference max dt=.00125, checked against .000625 on independent high-frequency states and extreme transfer parameters. Gate: relative L2 disagreement <=1e-5. Tests verify linear wave propagation against independently stepped RK4, finite gradients, diffusion mass/maximum principle/energy, and small-grid model run. No neural test rankings are used in these checks.

## Fixed matrix and training

24 cells = 2 PDEs x seeds {4101,4102,4103} x noise SD {0,.01} x {autonomous,query}. Each has 128 paired state snapshots, lags {.02,.05,.08,.12}; full batch Adam lr=.003, clip gradient norm 10, 5,000 updates, no learning-rate changes. Validation every 250 updates plus update 1, checkpoint chosen by GM of clean validation rollout MSE at horizons .6/1.2, lag .12, physical parameter 1. No test evaluation until training has ended. Initial, final, validation-best and exact 2,000/5,000-update checkpoints retained.

Training data/initialization seeds are paired across models. Training initial-state seed=2026091000+seed and noise seed=2026191000+seed. Val/test/high-frequency initial-state seeds=390901/390902/390903 with 32 states each; refinement seed390904, code-check seed390909.

Original smooth states contain Fourier frequencies1..4 with 1/k^2 coefficient scaling, individually normalized. Diffusion amplitude .25..75, mean -.2...2. Waves displacement amplitude .5..2, velocity amplitude .2..8, means -.2...2 independently drawn. The high-frequency pool changes frequencies to8..12 without 1/k^2 scaling, preserving the amplitude/mean ranges. No model sees high-frequency test states in training or validation.

## Endpoints and stopping

Primary separately for each PDE and noise: test physical parameter1, horizons1.2/2.4, lags.06/.12, GM of the four full-state MSEs, then GM of paired autonomous/query ratios across the three seeds. Material candidate gate: pooled ratio<=.90, at least2/3 seeds<=.90, no seed>1.05. This exploratory gate is not a significance test. The four primary settings are all reported, not only successes.

Secondary: horizons.6/1.2/2.4 and lags.06/.12/.24, low- and high-frequency tests; frozen-weight kappa transfer to.5/2 or c transfer to.5/1.5. Report relative L2, per-component wave error, diffusion mass drift/maximum-principle excess, physical energy increases, constitutive grid errors, nonuniform composition .24 versus .07+.17 at integration refinements1/2/4, and refined long rollout. Composition work is only approximately aligned by max-step policy; its refinement behavior must not be called an exact arithmetic identity. High-frequency and transfer outcomes do not rescue failed primary settings.

Report whether the best validation score continues improving in the final20% of updates. This is fixed-budget screening, not an assumption of convergence. No automatic training extension, seed replacement, or tuning on failure.

## Resource and artifacts

SCNet xhhgnormal, 1RTX3080+4CPU per task, OMP/MKL threads1, maximum4concurrent tasks, 45min cap per cell. One preparation/smoke job (60min cap) followed by24array tasks conditional on success. Maximum120,000formal updates. Unique canonical root /work/home/zenghang/semigroup-new-equations-20260910-r1; frozen source commit and SHA256, config, metrics, logs, cache, all endpoints and checkpoints retained. Account quota observed40GPU/200submissions; requested concurrency stays4. Preserve completed older roots.

Sources and prior-art boundaries: NEW_EQUATIONS_RATIONALE.md.
