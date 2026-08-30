
# Semigroup property: novelty audit and experiment gaps

Date: 2026-08-30

## Scope and conclusion

This is a targeted primary-source audit, not an exhaustive prior-art search. The project compares A, an autonomous/state-only generator whose time evolution is produced by integrating one shared vector field, with B, a query-time-conditioned model. It evaluates both prediction and an equal-work composition defect. The defensible conclusion is:

* The semigroup property itself, and learning variable-time evolution operators with semigroup awareness, are established ideas. In particular, Deep-OSG explicitly embeds the semigroup property in learning a family of evolution operators for autonomous ODEs and PDEs (Chen and Wu, 2023).
* The potentially novel contribution is a controlled *mechanism study*: hard autonomous generator versus time-conditioned flow-map baseline, parameter-matched controls, a non-autonomous reversal control, and partition/integrator stress tests that separate semigroup consistency from ordinary rollout MSE. This exact combination and its claimed causal interpretation are not established by the sources reviewed here.
* Current evidence supports a structural effect, not yet a universal accuracy claim: A has a much smaller equal-work composition defect, while Fisher formal MSE is approximately tied and does not meet the preregistered transfer decision rule. The non-autonomous negative control reverses the ranking, which is important boundary evidence.

Accordingly, the paper should claim a falsifiable hypothesis about when an autonomous generator is useful, not “we discovered semigroups for neural operators” or “A is always more accurate.”

## What is already common ground

For an autonomous well-posed evolution, the exact solution maps satisfy (S(0)=I) and (S(t+s)=S(t)S(s)). Neural ODEs make a shared continuous-time vector field an explicit modeling object; Chen et al. introduced Neural ODEs as continuous-depth models with an ODE-defined hidden state (Chen et al., 2018, https://arxiv.org/abs/1806.07366).

Neural operators learn maps between function spaces. FNO learns parametrized PDE solution operators using Fourier layers (Li et al., 2020, https://arxiv.org/abs/2010.08895), while DeepONet learns general nonlinear operators with branch/trunk networks (Lu et al., 2021, https://arxiv.org/abs/1910.03193). Neither paper, by itself, establishes the hard autonomous-generator versus query-time-conditioned comparison used here.

Koopman learning also has a long-standing operator-theoretic route to dynamics: a nonlinear state evolution is represented linearly after lifting observables. Deep neural dictionaries for Koopman operators were proposed by Yeung, Kundu, and Hodas (2017, https://arxiv.org/abs/1708.06850), and a continuous-time, stability-aware neural Koopman formulation was studied by Pan and Duraisamy (2019, https://arxiv.org/abs/1906.03663). These are adjacent in “operator structure for dynamics,” but they do not make the present A/B query-time ablation or equal-work partition test the central object.

## Directly adjacent work

The closest prior work is Deep-OSG: Junfeng Chen and Kailiang Wu, “Deep-OSG: Deep learning of operators in semigroup,” arXiv:2302.03358 (2023), https://arxiv.org/abs/2302.03358; the publisher record is https://www.sciencedirect.com/science/article/pii/S0021999123005934. The record identifies it as a Journal of Computational Physics article; this note intentionally does not infer volume, page, or article-number metadata from a related fixed-step paper. Deep-OSG explicitly learns a family of variable-time evolution operators for autonomous ODE/PDE systems, adds architecture/loss terms for semigroup structure, and reports accuracy and long-time stability benefits. This rules out novelty claims based solely on “using semigroup structure,” “variable lags,” or “composition consistency.”

Rotman et al., “Semi-supervised learning of partial differential operators and dynamical flows,” UAI 2023, PMLR 216:1785–1794, studies composition properties of PDE operators and dynamical flows (official paper: https://proceedings.mlr.press/v216/rotman23a/rotman23a.pdf). Its setting and semi-supervised objective differ from the present autonomous-versus-time-conditioned causal comparison.

SINGER (Feng et al., “Stochastic Network Graph Evolving Operator for High Dimensional PDEs,” ICLR 2025) explicitly targets evolution operators and states theoretical inheritance of graph topology, semigroup, and stability properties: https://proceedings.iclr.cc/paper_files/paper/2025/hash/5b288823575bb29654b0953a251e933b-Abstract-Conference.html. CFO (Hou, Huang, and Perdikaris, “CFO: Learning Continuous-Time PDE Dynamics via Flow-Matched Neural Operators,” ICLR 2026) learns a continuous-time neural operator/right-hand side and supports arbitrary temporal querying by ODE integration: https://proceedings.iclr.cc/paper_files/paper/2026/hash/8bfbf4ec87e1e331f0b1adc483b53b6b-Abstract-Conference.html; https://arxiv.org/abs/2512.05297. These further compress any claim that this project is the first continuous-time or semigroup-motivated PDE operator.

For gradient-flow priors, Phase-Field DeepONet (Li, Bazant, and Zhu, 2023) uses energy/variational structure and an explicit time-stepper for Allen–Cahn and Cahn–Hilliard: https://arxiv.org/abs/2302.13368. Energy-consistent Neural Operators (Tanaka et al., 2025, PMLR 258) impose conservation/dissipation behavior through an energy-based penalty: https://proceedings.mlr.press/v258/tanaka25a.html. Thus energy decrease, phase-field structure, and gradient-flow-aware operator learning are also prior art, distinct from the project’s composition-consistency mechanism.

These papers create a high bar: the manuscript must distinguish a hard continuous-time generator from a generic semigroup-aware family, and must explain why the equal-work metric is diagnostic rather than a theorem or a surrogate for test MSE.

## What may be new in the current project

The strongest novelty candidate is the *experimental decomposition*:

1. A is constrained to one time-homogeneous state-only generator; B receives query time. This directly tests whether time conditioning gives accuracy by representing different local maps, at the cost of cross-time composition.
2. Equal-work composition holds the physical horizon and numerical function-evaluation budget fixed. This makes the comparison less vulnerable to “one method got more solver work” confounding than a raw rollout comparison.
3. Composition is stress-tested over equal, unequal, and reversed partitions, plus 2/4/8/16 segments, rather than one hand-picked split.
4. Query-time fixed and shuffled controls test whether B actually uses its time input and whether that use is compatible with a single generator.
5. A capacity-matched A-wide control removes parameter-count as the simple explanation.
6. A non-autonomous PDE with explicit (r(t)) provides a negative control where the state-only autonomous hypothesis should fail and a time-aware model should win.

This is a potentially publishable mechanism/boundary contribution if it is replicated across independent seeds and at least one qualitatively different PDE, with predeclared metrics and no checkpoint selection on the test cache. It is not yet a claim of algorithmic novelty over Deep-OSG.

## Current evidence and its correct interpretation

The project’s existing records show: Allen–Cahn formal cells consistently favored A; Fisher formal cells show a large and consistent A advantage in equal-work defect but approximately tied rollout MSE, so the formal transfer criterion is not met; A-wide remains defect-superior after matching B’s parameter count; B-soft reduces defect modestly without a meaningful dose response; and the explicit non-autonomous control reverses the MSE ranking in favor of B. The latter is especially valuable because it guards against a blanket “autonomous is better” narrative.

The equal-work defect is a learned-map consistency diagnostic. It is not proof that the learned map is the exact PDE semigroup, and a lower defect does not imply lower one-step or long-horizon MSE without stability/approximation assumptions. A manuscript must report these quantities separately.

## Minimum falsification matrix

The smallest high-information next wave is:

* **Integrator control:** evaluate frozen A and B checkpoints with Euler, midpoint/RK2, and RK4, fixing physical horizon and function-evaluation budget. If A’s defect advantage disappears under a reasonable integrator family, the mechanism is a numerical artifact.
* **Lag–horizon phase diagram:** with a new exploratory cache (never the locked formal cache), evaluate frozen checkpoints over aligned lags and horizons extending beyond the formal range. If the gap does not grow with composition depth (T/\tau), reject the proposed “composition-depth amplification” explanation.
* **Second PDE:** use Burgers (transport–diffusion) or a linear advection–diffusion calibration. If A’s defect advantage appears only on reaction–diffusion and not on transport, narrow the claim. A simple linear calibration should ensure the evaluator itself does not favor A.
* **Non-autonomous replication:** repeat the explicit-time negative control for at least three seeds and report whether B wins directionally. If it does not, inspect state definition and conditioning before making a mechanism claim.
* **Capacity and soft controls:** retain parameter-matched A-wide and at least three fixed regularization strengths. If soft consistency reproduces all gains, the claimed importance of hard autonomous structure is weakened.

Primary decision rule: pre-register direction and aggregation before looking at the new test results; preserve all failed roots; keep formal locked inputs immutable; and never select a checkpoint using the exploratory phase diagram.

## Claims allowed / not allowed

Allowed after the minimum matrix passes: “In the tested autonomous PDEs, a hard state-only generator produces substantially more composition-consistent maps under equal-work stress; the effect survives parameter matching and is reduced or reversed in an explicitly non-autonomous control.”

Not allowed from the current evidence: “first semigroup neural operator,” “semigroup consistency guarantees better MSE,” “A universally outperforms B,” “the result transfers to all PDEs,” or “a low learned defect proves the exact semigroup law.”

## Primary sources consulted

* Chen, T. et al. (2018), Neural ODEs: https://arxiv.org/abs/1806.07366
* Li, Z. et al. (2020), Fourier Neural Operator: https://arxiv.org/abs/2010.08895
* Lu, L. et al. (2021), DeepONet: https://arxiv.org/abs/1910.03193
* Yeung, E. et al. (2017), deep Koopman representations: https://arxiv.org/abs/1708.06850
* Pan, S. and Duraisamy, K. (2019), physics-informed probabilistic Koopman embeddings: https://arxiv.org/abs/1906.03663
* Chen, J. and Wu, K. (2023), Deep-OSG: https://arxiv.org/abs/2302.03358; publisher record: https://www.sciencedirect.com/science/article/pii/S0021999123005934
* Rotman et al. (2023), “Semi-supervised learning of partial differential operators and dynamical flows,” UAI/PMLR 216:1785–1794: https://proceedings.mlr.press/v216/rotman23a/rotman23a.pdf
* Feng et al. (2025), SINGER, ICLR official proceedings: https://proceedings.iclr.cc/paper_files/paper/2025/hash/5b288823575bb29654b0953a251e933b-Abstract-Conference.html
* Hou, Huang, and Perdikaris (2026), CFO, ICLR official proceedings: https://proceedings.iclr.cc/paper_files/paper/2026/hash/8bfbf4ec87e1e331f0b1adc483b53b6b-Abstract-Conference.html; https://arxiv.org/abs/2512.05297
* Li, Bazant, and Zhu (2023), Phase-Field DeepONet: https://arxiv.org/abs/2302.13368
* Tanaka et al. (2025), Energy-consistent Neural Operators, PMLR 258: https://proceedings.mlr.press/v258/tanaka25a.html
