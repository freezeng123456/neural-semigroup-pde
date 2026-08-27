# Semigroup Attribution and Cross-PDE Transfer Protocol

**Status:** preregistered experimental protocol, written before the new
attribution runs. It separates a claim about autonomous semigroup structure
from the already observed benefit of the Allen--Cahn physical-energy anchor.

## 1. Scientific question and claim boundary

The current Allen--Cahn result is strong evidence that the bounded,
physics-anchored periodic latent flow is more accurate and more physically
monotone than its matched *no-anchor* latent control. It does **not** yet
identify the causal contribution of the semigroup property: both models are
autonomous latent flows, and the intervention is the fixed double-well energy
term.

The new question is narrower and falsifiable:

> Once state bounds, periodic locality, learned energy, the fixed physical
> energy term, positive mobility, RK4 budget, data, and optimization budget
> are matched, does one time-homogeneous generator improve long-horizon
> prediction and time-composition consistency relative to a model that may
> adapt its vector field to the requested time lag?

A positive result on Allen--Cahn and at least one genuinely different PDE
supports a broad semigroup-method paper. A negative or unstable result leads
to an Allen--Cahn-only paper whose claim is about a physics-anchored, bounded,
dissipative latent model, not a general semigroup advantage.

## 2. Allen--Cahn attribution matrix

| Label | Model | What it retains | What it removes/tests |
|---|---|---|---|
| **A** | `latent_physics_anchored_periodic` | Bounded logit coordinate, periodic decoded interaction, fixed double-well term, learned residual potential, positive local mobility, RK4 | One autonomous vector field `dz/dt=f(z)`; its exact continuous flow is a time-homogeneous semigroup. |
| **B** | `latent_physics_anchored_periodic_query_time` | Everything in A, including energy and positive mobility | Appends requested `tau` to the mobility stencil: `dz/dt=f(z;tau)`. Each fixed query remains energy-dissipative, but maps at different query times are not the flow of one generator. |
| **C** | Time-conditioned ResNet/FNO (secondary) | Same frozen data, optimizer budget and evaluation protocol | Standard direct-map reference. It is useful context but is not the causal A/B test because its coordinate, energy, and locality differ. |

The B control is initialized from A's mobility weights: state-stencil columns
are copied exactly and the newly introduced `tau` column starts at zero. Thus
it initially computes the same vector field as A and adds only 64 parameters
(0.70% at the 9,093-parameter Allen--Cahn setting).

## 3. Fixed protocol for the first screen

The first formal screen uses a fresh variable-lag cache, not the earlier
fixed-`tau=0.1` study.

* **PDE:** 1D periodic Allen--Cahn, `N=64`, `epsilon=0.1`, reference step
  `0.005`.
* **Data:** 1,000 training trajectories and 50 validation trajectories. Each
  A/B pair reads the identical frozen cache.
* **Training lags:** `0.025, 0.05, 0.10, 0.20`, balanced in the cache.
* **Unseen-lag tests:** `0.075` and `0.15`; neither is a training lag.
* **Long-horizon tests:** physical horizons `1.2`, `2.4`, and `4.8`, evaluated
  by repeated application without using test observations to choose a
  checkpoint.
* **Budget:** 100 epochs, batch size 64, Adam `1e-3`, validation every five
  epochs, the same training/data-order seed inside each A/B pair, and 30 RK4
  substeps per query.
* **Losses:** optional rollout, trajectory, learned-energy, bounds and V-value
  losses are zero for both A and B. The first screen uses `beta_v_floor=0` to
  match the successful prior anchor runs; it therefore does not claim global
  latent coercivity. A positive result is followed by a floor robustness
  check, not silently generalized to it.
* **Independent seeds:** `42`, `137`, and `2718`.

Every formal cell needs a unique root, run card, source archive hash,
frozen-cache source/local before-and-after hashes, selected-checkpoint hash,
separate locked-test cache, and a result receipt. Local smoke numbers are
never pooled with formal results.

## 4. Endpoints and advance decision rule

For each seed and test condition define `R = MSE(A) / MSE(B)`. The aggregate
accuracy statistic is the geometric mean of `R` over the predeclared seeds and
horizons; all individual ratios are reported alongside it. Values below one
favor the autonomous model.

The required endpoints are:

1. rollout MSE and relative L2 at each lag and horizon;
2. physical-energy monotonic fraction and positive-increment statistics;
3. bound violations;
4. production and equal-work composition defects;
5. RK4-substep convergence, so finite-integrator behavior is not claimed as
   an exact numerical-map semigroup; and
6. wall-clock throughput and parameter count.

**Advance criterion for the broad route.** In Allen--Cahn, A must beat B by at
least 10% in aggregate long-horizon MSE (`R <= 0.90`), have the same direction
of benefit in at least two of three seeds, not worsen physical-energy
monotonicity by more than 0.02, and have lower equal-work composition defect.
It must then meet the same accuracy and composition criteria on one transfer
PDE. Otherwise, the work is reported as an Allen--Cahn physics-anchor result,
with the semigroup result called inconclusive rather than forced into a broad
claim.

## 5. Cross-PDE sequence and model-class boundaries

### 5.1 Fisher--KPP / logistic reaction--diffusion — first transfer

Fisher--KPP is the nearest transfer: periodic, scalar, bounded in `[0,1]`,
reaction--diffusion rather than phase separation, and expressible as an L2
gradient flow for a diffusion-plus-logistic-reaction energy. The code exposes:

* **A-Fisher:** `LatentSemigroupNet`, an autonomous learned gradient flow;
* **B-Fisher:** `QueryTimeConditionedLatentFlow`, with the same learned energy,
  interactions, positive mobility, and RK4 integration but query-time mobility
  conditioning.

Fisher uses the same variable-lag schedule, paired frozen caches, three seeds,
and long-horizon/unseen-lag diagnostics. Its direct ResNet/FNO results are
secondary context, not substitutes for the A/B attribution. The legacy
Fisher runner was corrected so no model gets an unmatched bound loss during
this screen.

### 5.2 Cahn--Hilliard — scientific transfer after Fisher

Cahn--Hilliard is mass-conserving and has an H^{-1} gradient flow, not the
diagonal-mobility L2 flow used here. A valid extension must replace `K(z)` by a
conservative positive operator such as `-D^T M(z) D` and make mass error a
primary endpoint. Applying the current diagonal model unchanged would not be a
meaningful transfer test.

### 5.3 Burgers — a deliberate boundary case

Viscous Burgers has transport plus diffusion and does not fit the current
purely dissipative gradient-flow architecture. A valid experiment needs a
conservative/skew transport generator or explicitly split transport and
dissipation. The legacy Burgers script has unequal losses and incomplete
provenance; it is excluded from formal evidence. A failure there would define
the method's current domain of validity, not refute autonomous semigroups.

### 5.4 Later targets

Kuramoto--Sivashinsky and 2D incompressible Navier--Stokes are appropriate only
after the two results above: they test chaotic/non-gradient dynamics and
higher-dimensional fields, but require different structure-preserving
generators and much larger compute.

## 6. Relation to prior work

The protocol is stricter than showing merely that an autonomous neural ODE can
work. Deep-OSG explicitly learns semigroup operators over variable increments,
while Continuous Flow Operators learn continuous PDE velocity fields and
benchmark Burgers and diffusion--reaction equations. Energy-aware phase-field
operators already address Allen--Cahn/Cahn--Hilliard free-energy structure, so
an energy benefit alone cannot establish the semigroup claim. The A/B test
holds energy structure fixed and tests the additional time-homogeneity claim.

* [Deep-OSG: Deep Operator Semigroup Networks](https://arxiv.org/abs/2302.03358)
* [Continuous Flow Operators](https://arxiv.org/abs/2512.05297)
* [Phase-Field DeepONet](https://arxiv.org/abs/2302.13368)
* [Energy-consistent Neural Operators](https://proceedings.mlr.press/v258/tanaka25a.html)

## 7. Execution order

1. Run and review local A/B smoke tests for Allen--Cahn and Fisher--KPP.
2. Publish the source change and prepare a formal T4 Allen--Cahn variable-lag
   paired cache plus seed-42 A/B screen.
3. If the seed-42 artifacts are complete and no numerical anomaly appears,
   run seeds 137 and 2718 sequentially. Do not tune from locked tests.
4. Run the Fisher A/B screen only after the Allen attribution result is
   recovered, so transfer data cannot quietly change the Allen protocol.
5. Apply the advance criterion before allocating A100-scale compute and choose
   either the broad semigroup paper or the focused Allen--Cahn paper.
