# Evidence for learning PDE evolutions as flows or semigroups

**Question.** What evidence supports learning an autonomous continuous-time flow, semigroup, or operator semigroup for PDE evolution, relative to (i) a discrete one-step Markov/residual map, (ii) a direct time-conditioned map `(u0,t) -> ut`, (iii) a non-autonomous latent ODE, and (iv) a physics-energy model without autonomous semigroup consistency?

**Scope.** This targeted reading uses primary papers and official proceedings/project pages. It finds substantial evidence for continuous-time *parameterization* (irregular-time queries, shared dynamics, solver-based rollout), and for long-horizon problems being sensitive to autoregressive error. It does **not** establish that an autonomous semigroup constraint, by itself and at matched capacity/compute, improves Allen–Cahn prediction. That narrower claim requires the ablation below.

## Short conclusion

The defensible rationale is inductive bias, not a generic accuracy claim. An autonomous generator/flow represents one time-homogeneous law whose exact flow obeys `S_(t+s) = S_t o S_s`; it exposes a testable consistency property and supports arbitrary time increments. Neural ODE and latent-flow papers show usefulness for irregular sampling, shared long-horizon dynamics, and reduced-order modeling. Neural-operator papers show that direct maps and autoregressive one-step maps are strong, scalable baselines, but autoregression can accumulate error and fixed-step models do not automatically answer new time increments.

The strongest evidence for this Allen–Cahn design would be an improvement that survives equal parameters, equal vector-field/RK work, the same encoder/decoder, unseen time increments, and comparison with a direct time-conditioned model that receives the same fixed double-well information.  In the current N=128 protocol, `alpha_rollout=0` and `alpha_energy=0`: the semigroup prior comes from the autonomous generator, not from an explicit semigroup or physical-energy training penalty.  A paper should therefore not describe “removing the semigroup loss” as the decisive current ablation.

## Most relevant primary works

| Work | Evidence | Relevance and limitation |
|---|---|---|
| Chen et al., *Neural Ordinary Differential Equations* (NeurIPS 2018), [arXiv:1806.07366](https://arxiv.org/abs/1806.07366), [DOI](https://doi.org/10.48550/arXiv.1806.07366) | Learns a continuous vector field and evaluates states by an ODE solver; emphasizes adaptive computation and continuous-depth dynamics. | Direct precedent for learning a generator rather than a one-step map. It is finite-dimensional, not a PDE semigroup theorem; solver error and function-evaluation cost are part of the method. |
| Chen & Wu, *Deep-OSG: Deep Learning of Operators in Semigroup* (2023), [arXiv:2302.03358](https://arxiv.org/abs/2302.03358) | Learns variable-lag evolution operators and explicitly embeds the semigroup law using a time-aware residual architecture and a semigroup-informed regularizer; reports long-time prediction and robustness experiments on ODEs/PDEs. | The closest direct semigroup-learning precedent. Its claimed benefits are encouraging but rely on its own data organization, losses, and benchmarks; our paper needs its own matched Allen--Cahn counterfactual rather than importing its conclusion. |
| Li et al., *Fourier Neural Operator for Parametric PDEs* (ICLR 2021), [arXiv:2010.08895](https://arxiv.org/abs/2010.08895), [OpenReview](https://openreview.net/forum?id=c8P9NQVtmnO) | Learns a mesh-discretization-independent operator and reports strong PDE benchmarks. | Essential direct/operator baseline. Standard evolution use is usually a learned finite-step transition or direct solution operator, not an autonomous semigroup constraint. |
| Li et al., *Physics-Informed Neural Operator* (2021), [arXiv:2111.03794](https://arxiv.org/abs/2111.03794) | Combines coarse data with high-resolution PDE residuals; reports data efficiency and zero-shot super-resolution. | Control for “physics helps” versus “semigroup helps.” Its gains do not evidence semigroup consistency. |
| Tanaka et al., *Energy-consistent Neural Operators for Hamiltonian and Dissipative Partial Differential Equations* (AISTATS 2025), [PMLR](https://proceedings.mlr.press/v258/tanaka25a.html) | Biases direct neural operators toward energy conservation/dissipation through a training penalty and reports improved solution prediction. | A particularly useful physics-without-semigroup comparison: energy-aware behavior can be obtained without an autonomous latent flow, so energy improvement alone cannot establish a semigroup advantage. |
| Lu et al., *DeepONet* (Nature Machine Intelligence 2021), [arXiv:1910.03193](https://arxiv.org/abs/1910.03193), [DOI](https://doi.org/10.1038/s42256-021-00302-5) | Branch/trunk operator learning approximates mappings between functions and outputs, including PDE solution operators. | Natural direct `(u0,t)->ut` comparator. It has no composition guarantee unless explicitly trained or imposed. |
| Long et al., *PDE-Net 2.0* (ICML 2019), [PMLR](https://proceedings.mlr.press/v97/long19a.html), [arXiv:1812.04426](https://arxiv.org/abs/1812.04426) | Learns interpretable differential operators/nonlinear terms in a time-marching architecture; demonstrates prediction and equation discovery. | Structured discrete physics-shaped comparator. It can work without an autonomous latent flow, though its stencil parameterization is not automatically capacity-matched. |
| Fresca et al., *Neural Latent Dynamics Models for Reduced Order Modelling* (2021), [arXiv:2105.05053](https://arxiv.org/abs/2105.05053), [OpenReview](https://openreview.net/forum?id=Yk_I37Ca8Q) | Uses encoder–latent neural ODE–decoder models for deterministic reduced-order dynamics and separates representation learning from latent dynamics learning. | Closest architectural precedent. It does not isolate autonomy/semigroup law from latent compression or ODE regularization, and is not an Allen–Cahn energy ablation. |
| Rubanova et al., *Latent ODEs for Irregularly-Sampled Time Series* (NeurIPS 2019), [arXiv](https://arxiv.org/abs/1907.03907), [OpenReview](https://openreview.net/forum?id=HygCYNSlLB) | Shows why continuous latent dynamics help with irregular observations. | Supports variable-time evaluation as a meaningful benefit. It is not PDE operator learning, and its stochastic/inference setup differs from deterministic autonomous PDE flow. |
| Brandstetter et al., *Message Passing Neural PDE Solvers* (ICLR 2022), [OpenReview](https://openreview.net/forum?id=vBbBwZwp2P) | Learns PDE evolution with graph message passing and autoregressive rollout; evaluates long-horizon accuracy. | Strong discrete Markov baseline. A proposed model must beat a well-tuned one-step model, not only a weak residual net. |
| Morel et al., *DISCO: learning to DISCover an evolution Operator for multi-physics-agnostic prediction*, [OpenReview](https://openreview.net/forum?id=6EZ3MDDf6p) | Infers an evolution operator from a short trajectory and integrates a learned neural PDE/operator for future prediction across physics. | Relevant flow/operator evidence and non-autonomous/task-conditioned comparator. Conditioning dynamics on trajectory information is not the same as one shared autonomous semigroup. |
| Hou, Huang & Perdikaris, *CFO: Learning Continuous-Time PDE Dynamics via Flow-Matched Neural Operators* (2025), [arXiv:2512.05297](https://arxiv.org/abs/2512.05297) | Learns a neural-operator velocity field and integrates it; reports arbitrary temporal querying, irregular training grids, and long-horizon results on Burgers, diffusion-reaction, and shallow water. | Strong recent motivation for continuous-time PDE dynamics, especially unseen `tau`. It does not isolate autonomous semigroup consistency from direct/one-step alternatives, and its flow-matching objective differs from dissipative latent gradient flow. |

## What the papers support—and do not support

1. **Continuous time is operationally useful.** Neural ODE/latent ODE and CFO support arbitrary or irregular time queries and shared temporal dynamics. This supports testing unseen increments, not assuming a gain at the training step.
2. **Long-horizon rollout is a real failure mode.** FNO/PINO and message-passing solvers are strong alternatives, while CFO motivates continuous-time integration partly through accumulated error and temporal-grid restrictions. Measure error versus physical horizon and equal-work composition error.
3. **Physics is not semigroup structure.** PINO, PDE-Net, and energy/PDE residual training can improve fidelity without implying `S_(t+s)=S_t o S_s`. For this project, retain the same *fixed double-well information* (for example as an energy/force feature or an otherwise matched fixed module) in the direct no-semigroup ablation. The current protocol has no physical-energy loss to retain.
4. **Latent ODE evidence is confounded.** Wins may come from compression, parameter sharing, smoother interpolation, or optimization. Match encoder/decoder and budget.
5. **Theory is conditional.** Autonomous ODE theory gives a flow composition law for a sufficiently well-posed vector field; operator universal approximation gives direct-map expressivity. Neither says a finite-step numerical implementation is an exact PDE semigroup or more accurate at finite compute.

## Minimal decisive Allen–Cahn matrix

Keep fixed: data/split, encoder/decoder, latent width, optimizer updates, training pairs, seeds, physical-energy definition, admissible-state handling, horizon, and parameter count within a small tolerance. Report relative L2 error, rollout error versus physical time, physical-energy monotonicity, bound violations, latent/vector-field norms, composition defect, wall time, memory, and exact vector-field/RK evaluations.

| ID | Model | Isolated change | Required test |
|---|---|---|---|
| A | Autonomous dissipative latent flow with the fixed Allen--Cahn double-well term | Full proposed design: `S+P` | Nominal, unseen interpolation, extrapolated `tau`; direct-vs-composed `S_(2tau)`; long rollout. |
| B | Current autonomous latent flow without the fixed double-well term | Removes physics anchor only: `S-P` | This is the existing matched control; it cannot isolate the semigroup contribution. |
| C | Time-conditioned direct map with the same bounded state representation and fixed double-well force/energy feature | Key genuinely non-semigroup counterfactual: `D+P` | Train on multiple increments; evaluate the direct map at 1.2/2.4/4.8 and measure its composition defect without using that defect for model selection. |
| D | Matched time-conditioned direct map without the fixed double-well information | `D-P` | Completes the 2x2 factorial design, separating physics information from autonomous semigroup structure. |
| E | Same representation and physical anchor, non-autonomous latent ODE `dz/dt=f(z,t)` | Removes time homogeneity while retaining continuous integration | Match vector-field capacity and solver work; test different decompositions from the same initial state. |
| F | Physics-conditioned discrete one-step residual map, recursively rolled out | Removes the continuous generator but retains a discrete iterated map | Match parameters and updates. Label it accurately: repeated application defines a discrete semigroup at integer steps, so it tests continuous-time generation, not the full absence of composition structure. |

Run A–F on three seeds after a single-seed smoke screen. The primary endpoint is paired fixed-horizon rollout relative L2/MSE at matched compute; secondary endpoints are unseen-time performance and composition defect. The indispensable comparison is **A versus C**: both have the same fixed Allen--Cahn information, but only A is generated by one autonomous flow. A semigroup claim should require A to beat C and E, while B/D identify whether the double-well prior itself explains the gain. If A only wins on composition defect or interpolation, make that narrower claim. A tie on prediction with materially better defect, energy monotonicity, or long-horizon stability is a legitimate structural result, not universal accuracy superiority.

## Strong reviewer objections

- **Unfair comparison:** match parameters, updates, examples, solver/vector-field calls, horizon, memory, and wall time; report fixed-step and equal-work metrics.
- **Latent bottleneck confound:** use the same encoder/decoder and latent dimension in B–E.
- **Energy confound:** include the `D+P` direct map in C and report physical and learned energy separately.
- **RK4 made the semigroup:** RK4 maps are not exact semigroups; sweep solver steps and distinguish continuous-flow from numerical claims.
- **Time conditioning is stronger:** give C the same time range/budget and separate interpolation from extrapolation.
- **Autonomy was assumed by construction:** make it empirical; D may fit finite data better while A should be more consistent across decompositions.
- **Energy decay is only surrogate decay:** report physical Allen–Cahn free energy, monotone-transition fraction, and positive-increment magnitude.
- **Reference/seed artifacts:** use grid/time-step refinement, three seeds, confidence intervals, and fixed data seed before interpreting small differences.

## Proposed claim wording

> On the Allen–Cahn benchmark, under matched state representation, fixed double-well information, training budget, and integration work, the autonomous dissipative latent-flow model [does/does not] improve long-horizon rollout error relative to a direct time-conditioned no-semigroup map and a non-autonomous continuous-time map. Its distinctive empirical benefit is [smaller composition defect / better unseen-time interpolation / improved physical-energy behavior]. These results support a conditional structural advantage for this benchmark, not a general theorem that semigroup models outperform neural operators.

## Bibliography

- Chen et al. (2018), *Neural Ordinary Differential Equations*. [arXiv](https://arxiv.org/abs/1806.07366)
- Chen & Wu (2023), *Deep-OSG: Deep Learning of Operators in Semigroup*. [arXiv](https://arxiv.org/abs/2302.03358)
- Long et al. (2019), *PDE-Net 2.0*. [PMLR](https://proceedings.mlr.press/v97/long19a.html)
- Rubanova et al. (2019), *Latent ODEs for Irregularly-Sampled Time Series*. [OpenReview](https://openreview.net/forum?id=HygCYNSlLB)
- Li et al. (2021), *Fourier Neural Operator*. [OpenReview](https://openreview.net/forum?id=c8P9NQVtmnO)
- Li et al. (2021), *Physics-Informed Neural Operator*. [arXiv](https://arxiv.org/abs/2111.03794)
- Tanaka et al. (2025), *Energy-consistent Neural Operators for Hamiltonian and Dissipative Partial Differential Equations*. [PMLR](https://proceedings.mlr.press/v258/tanaka25a.html)
- Lu et al. (2021), *DeepONet*. [DOI](https://doi.org/10.1038/s42256-021-00302-5)
- Fresca et al. (2021), *Neural Latent Dynamics Models*. [OpenReview](https://openreview.net/forum?id=Yk_I37Ca8Q)
- Brandstetter et al. (2022), *Message Passing Neural PDE Solvers*. [OpenReview](https://openreview.net/forum?id=vBbBwZwp2P)
- Morel et al., *DISCO*. [OpenReview](https://openreview.net/forum?id=6EZ3MDDf6p)
- Hou, Huang & Perdikaris (2025), *CFO*. [arXiv](https://arxiv.org/abs/2512.05297)
