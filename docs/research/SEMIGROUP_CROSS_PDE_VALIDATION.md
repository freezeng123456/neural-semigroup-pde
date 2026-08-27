# Cross-PDE Validation Plan for Autonomous Neural Semigroups

## Executive decision

The project should run two tracks in parallel:

1. **Semigroup attribution track:** hold physics information and model capacity fixed, then compare an autonomous continuous generator against direct time-conditioned and non-autonomous alternatives. This is the only track that can establish that the semigroup/autonomy itself helps.
2. **PDE-transfer track:** apply the same semigroup backbone to one nearby dissipative PDE and one structurally different PDE. This tests whether the idea transfers beyond Allen--Cahn, while exposing the boundary of the current gradient-flow parameterization.

The recommended first transfer pair is **Fisher--KPP/reaction--diffusion** and **Cahn--Hilliard**. Viscous Burgers is the next boundary test. Kuramoto--Sivashinsky and 2D Navier--Stokes should be deferred until the attribution protocol and mixed transport--dissipation generator are working.

## What the literature actually validates

For an autonomous evolution

\[
 \partial_t u=F(u), \qquad S_tu_0=u(t),
\]

the exact solution operators satisfy `S_(t+s) = S_t o S_s`. A learned model must therefore be tested at three distinct levels:

* **accuracy:** does it predict the reference trajectory?
* **composition:** does a direct prediction at `t+s` agree with two predictions at `s` and `t`?
* **generalization in time:** does it work for time lags and decompositions not used in training?

Deep-OSG is the closest direct precedent. It learns a family of variable-lag evolution operators and explicitly embeds semigroup structure in the architecture and loss; it reports accuracy, robustness and long-time benefits across ODE/PDE examples. Its existence means this project must not claim the first neural semigroup. The relevant comparison is whether an **autonomous latent generator with structure-preserving physics** provides a different or stronger inductive bias than a generic semigroup-aware operator. [Deep-OSG](https://arxiv.org/abs/2302.03358)

Continuous-time operator work gives a second comparison class. CFO learns a continuous-time PDE velocity field with flow matching and integrates it at inference; it reports arbitrary temporal querying and long-horizon experiments on Burgers, diffusion-reaction and shallow water. This supports comparing generator-based models against direct autoregressive and time-conditioned operators, but CFO is not by itself a proof of the semigroup advantage. [CFO](https://arxiv.org/abs/2512.05297)

Energy-aware operator papers establish that energy behavior can be improved without an autonomous semigroup. Energy-consistent Neural Operators use structured skew/dissipative operators and test Cahn--Hilliard; ECO studies energy-constrained operator learning for dissipative systems. Therefore, an energy improvement alone cannot be attributed to semigroup structure. [Energy-consistent Neural Operators](https://proceedings.mlr.press/v258/tanaka25a.html), [ECO](https://proceedings.mlr.press/v331/goertzen26a.html)

Phase-field operator learning is also a direct neighboring line: Phase-Field DeepONet studies Allen--Cahn and Cahn--Hilliard with free-energy/gradient-flow information. This makes Cahn--Hilliard a credible transfer benchmark, but it also means the paper must compare the time-evolution parameterization, not merely claim the first physics-aware phase-field learner. [Phase-Field DeepONet](https://arxiv.org/abs/2302.13368)

The literature therefore supports the following claim boundary:

> The novelty target is not “semigroup exists” or “energy helps.” It is whether an autonomous, structure-preserving latent generator gives a measurable advantage over equally informed direct and non-autonomous alternatives across PDE classes.

## Attribution protocol: the minimum experiment that isolates semigroup benefit

For every PDE, use the same frozen trajectories, source/validation/test splits, parameter budget, optimizer, training updates, checkpoint-selection rule and inference budget. Use four model cells:

| Cell | Time parameterization | Physics information |
|---|---|---|
| A | autonomous latent generator `dz/dt=f(z)` | full matched physics anchor |
| B | autonomous latent generator | no physics anchor |
| C | direct time-conditioned map `G(u,t)` or one-step map | full matched physics information |
| D | direct time-conditioned map | no physics anchor |

The decisive comparison is **A versus C**. B versus A only measures the physics anchor. C versus D measures physics information in a non-semigroup model. If A wins over C with matched physics and capacity, the gain can be attributed to the autonomous semigroup bias. Add a fifth cell when feasible:

* **E:** non-autonomous latent ODE `dz/dt=f(z,t)` with the same latent coordinates and comparable parameter count.

A versus E isolates time-homogeneity/autonomy from continuous-time integration. A versus a fixed-step one-step baseline isolates the benefit of a single generator from repeated learned maps.

Required metrics:

* held-out MSE and relative `L2` at short, medium and long horizons;
* composition defect `||G(x,t+s)-G(G(x,s),t)||`, including unseen decompositions;
* time-lag interpolation error for lags absent from training;
* rollout stability and failure rate;
* physical invariant/monotonicity diagnostics;
* training cost, inference wall time and number of vector-field/function evaluations.

Do not use the test set to select a checkpoint, tune a time lag or choose a decomposition. Report per-trajectory distributions, not only averages.

## Track 1: semigroup attribution on Allen--Cahn

Keep the current periodic 1D Allen--Cahn benchmark as the anchor task. The current A/B result is strong evidence for the double-well physics anchor, but does not isolate semigroup because both models are autonomous.

Run C and E with:

* the same bounded physical coordinate and parameter count;
* the same fixed double-well term for A and C;
* the same training trajectories and locked 500-trajectory test set;
* the same three horizons and checkpoint protocol;
* explicit unseen-lag and composition tests.

### Go criterion for a semigroup claim

Proceed with a broad semigroup-centered paper only if, on Allen--Cahn and at least one transfer PDE:

1. A improves over C on the pre-registered long-horizon metric by at least 10% relative, or wins on a pre-registered composite score;
2. A has lower composition defect and better unseen-lag interpolation than C;
3. the direction holds for at least 3 independent seeds and is not explained by a larger effective function-evaluation budget;
4. the advantage remains when A and C receive the same physics information.

If A beats B but not C, the safe paper is an Allen--Cahn physics-anchor paper. If A beats C only at long horizon but not short horizon, the claim should be “semigroup improves long-time consistency,” not universal accuracy.

## Track 2: PDE transfer matrix

### 1. Fisher--KPP / reaction--diffusion: nearest-neighbor transfer

Representative form:

\[
 u_t=D u_{xx}+r u(1-u/K),
\]

with periodic or homogeneous boundary conditions and a positive invariant interval. For autonomous fixed coefficients, the forward solution is a semigroup. This is the cleanest neighboring test because it preserves local reaction--diffusion structure while changing the reaction potential and state constraint.

**Best structural family:** bounded positive-coordinate autonomous flow, with diffusion stencil plus a reaction module. Use a Fisher reaction anchor or a learned reaction residual; do not reuse the Allen--Cahn double well.

**Minimal valid test:** A/B/C/E on the same frozen initial-condition ensemble; train on several lags, test on unseen lags; evaluate short/medium/long rollout, positivity violation, composition defect and front speed. Match parameter count and integration budget.

**Expected failure modes:** sigmoid saturation near `u=0` or `K`; front translation error; reaction-diffusion stiffness; a direct map may interpolate time better on the short training interval; an unconstrained model may obtain lower MSE while violating positivity.

**Decision:** strong candidate for the first transfer. If A improves composition and long-horizon stability while preserving positivity, the semigroup claim has credible in-class replication.

### 2. Cahn--Hilliard: conservation-gradient-flow transfer

Representative form:

\[
 u_t=\nabla\cdot(M\nabla \mu),\qquad
 \mu=-\epsilon^2\Delta u+W'(u).
\]

The deterministic autonomous solution defines a forward semigroup on an appropriate state space. The equation conserves mass and dissipates free energy. It is not an ordinary `L2` gradient flow; it is an `H^{-1}`-type gradient flow with a fourth-order spatial effect.

**Best structural family:** mass-conserving divergence-form generator, e.g. `dz/dt = C_theta(z) - div(M_theta grad(mu_theta))`, with an explicit zero-mean projection or conserved-mass parameterization. Retain the autonomous generator, but replace the Allen--Cahn mobility/anchor with chemical potential, mobility and mass constraints.

**Minimal valid test:** A/B/C against a direct time-conditioned phase-field operator, with identical free-energy information; report mass drift, energy dissipation, composition defect, long-time phase separation statistics and MSE. Test multiple initial masses, not only one mass level.

**Expected failure modes:** mass drift from numerical integration; stiffness from the fourth-order term; over-smoothed phase interfaces; a direct operator may look competitive at short horizon; a naive positive mobility does not enforce conservation.

**Decision:** strongest second PDE. It tests whether the same semigroup idea survives when the physical geometry changes from `L2` gradient flow to mass-conserving `H^{-1}` flow. Failure of the current model here is informative and should trigger a structural extension, not be counted as failure of semigroup itself.

### 3. Viscous Burgers / advection--diffusion: transport boundary test

Representative form:

\[
 u_t+u u_x=\nu u_{xx}.
\]

For fixed `nu` and autonomous boundary conditions, the forward evolution is semigroup-like. But it is not a pure gradient flow: transport redistributes energy while viscosity dissipates it.

**Best structural family:** split generator
\[
 \dot z=A_\theta(z)-K_\theta(z)\nabla\Psi_\theta(z),
\]
where `A` models transport and the second term models dissipation. Use conservative/flux-form spatial interactions where possible.

**Minimal valid test:** compare A (split autonomous), C (time-conditioned direct map), E (non-autonomous latent ODE), and a one-step conservative baseline. Report shock/front phase error, MSE, composition defect, energy behavior, mass/mean drift and stability as `nu` decreases.

**Expected failure modes:** artificial diffusion; phase error dominates MSE; non-smooth gradients make adjoint training unstable; the pure gradient-flow model fails despite semigroup being valid; numerical solver tolerances change the apparent composition defect.

**Decision:** use as a boundary test only after Fisher--KPP succeeds. A negative result from the pure gradient-flow architecture is expected and should motivate the split generator.

### 4. Kuramoto--Sivashinsky: stiff dissipative semigroup

The standard periodic Kuramoto--Sivashinsky equation has autonomous fourth-order dynamics with linear instability and nonlinear saturation. Its forward dynamics are semigroup-like, but the system is chaotic and stiff.

**Best structural family:** autonomous spectral or local generator with explicit linear stiff operator treatment plus learned nonlinear residual; optionally a split dissipative/transport generator.

**Minimal valid test:** use short-horizon state MSE plus statistical/attractor diagnostics; test composition defect, Lyapunov/energy proxies, spectrum and rollout failure probability. Do not rely on pointwise long-horizon MSE alone because chaos makes trajectory-wise accuracy ill-conditioned.

**Expected failure modes:** exponential error growth, phase drift, stiffness, solver-step dependence, and misleadingly good short-horizon fit with failed invariant statistics.

**Decision:** relevant for a mature broad paper, not the first transfer. It tests whether semigroup consistency helps before pointwise chaos destroys predictability.

### 5. 2D Navier--Stokes: high-value but high-risk extension

For fixed forcing, viscosity and boundary conditions, the forward flow is autonomous and semigroup-like in the appropriate function space, but the practical regime matters. The generator combines transport, pressure projection, incompressibility and viscous dissipation.

**Best structural family:** constrained divergence-free transport--dissipation generator, preferably with a projection or stream-function representation, not a scalar Allen--Cahn-style potential.

**Minimal valid test:** compare autonomous generator, direct time-conditioned operator and one-step baseline on the same initial-condition ensemble; report velocity MSE, vorticity error, divergence violation, kinetic-energy spectrum, enstrophy, composition defect and long-time statistics. Match spatial resolution, forcing and Reynolds-number regime.

**Expected failure modes:** divergence drift, pressure/aliasing errors, turbulence sensitivity, huge evaluation cost, and pointwise MSE becoming uninformative at long horizon.

**Decision:** do not make 2D Navier--Stokes a go/no-go prerequisite. It is appropriate only after the lower-risk tests establish attribution and after a structure-specific generator is implemented.

## Recommended execution order

1. **Allen--Cahn attribution:** add matched direct time-conditioned and non-autonomous controls. This is mandatory before a semigroup-centered claim.
2. **Fisher--KPP:** nearest reaction--diffusion transfer with positivity and front-speed diagnostics.
3. **Cahn--Hilliard:** conservation-gradient-flow transfer with mass and energy diagnostics.
4. **Viscous Burgers:** transport boundary test using a split generator.
5. **Kuramoto--Sivashinsky:** only if the project wants a stiff/chaotic benchmark.
6. **2D Navier--Stokes:** a separate ambitious extension, not part of the minimal paper.

## Go/no-go rule for the paper

### Broad semigroup paper: GO only if

* Allen--Cahn A beats the matched physics-informed direct-map C on long-horizon accuracy and composition defect;
* the direction replicates on Fisher--KPP or Cahn--Hilliard across at least 3 seeds;
* the gain survives matched parameter count, data, optimizer and function-evaluation budget;
* the structural model is adapted correctly to each PDE's invariants rather than forcing every equation into an Allen--Cahn gradient flow;
* all failures and non-applicable PDE classes are reported.

### Allen--Cahn-only paper: GO if

* A/B remains the strongest result but A/C is neutral or inconclusive;
* the physics anchor yields robust locked-test accuracy and energy behavior;
* the paper explicitly states that semigroup causality was not isolated;
* claims are restricted to a bounded physics-anchored autonomous latent-flow solver for the stated Allen--Cahn benchmark.

### Stop conditions

Stop broad expansion and write the Allen--Cahn paper if:

* A does not beat C after matched tuning and at least 3 seeds;
* the composition defect is not better than C;
* transfer gains appear only when the autonomous model receives more compute or more temporal supervision;
* the model fails the invariant appropriate to a PDE despite lower MSE.

## Practical interpretation

The project should not ask whether “semigroup works for every PDE.” The correct question is whether the **autonomous generator inductive bias** improves prediction when the PDE is autonomous, and whether its internal structure is adapted to the PDE's geometry. Allen--Cahn and Fisher--KPP test the same dissipative family; Cahn--Hilliard tests conservation; Burgers tests transport; Kuramoto--Sivashinsky and Navier--Stokes test stiffness/chaos and mixed structure.

The broad-paper claim is therefore conditional and falsifiable. If the first two transfers fail the A-versus-C test, the scientifically correct outcome is an Allen--Cahn paper with a clear negative result about semigroup attribution—not an overextended universal claim.

## Primary sources

* Chen & Wu, “Deep-OSG: Deep Learning of Operators in Semigroup,” arXiv:2302.03358. https://arxiv.org/abs/2302.03358
* Hou, Huang & Perdikaris, “CFO: Learning Continuous-Time PDE Dynamics via Flow-Matched Neural Operators,” arXiv:2512.05297 / ICLR 2026. https://arxiv.org/abs/2512.05297
* Tanaka et al., “Energy-consistent Neural Operators,” Proceedings of Machine Learning Research 258 (2025). https://proceedings.mlr.press/v258/tanaka25a.html
* Goertzen et al., “ECO: Energy-Constrained Operator Learning,” Proceedings of Machine Learning Research 331 (2026). https://proceedings.mlr.press/v331/goertzen26a.html
* “Phase-Field DeepONet: A Deep Operator Network for Phase-Field Simulations,” arXiv:2302.13368. https://arxiv.org/abs/2302.13368
* Li et al., “Fourier Neural Operator for Parametric Partial Differential Equations,” ICLR 2021. https://openreview.net/forum?id=c8P9NQVtmnO

