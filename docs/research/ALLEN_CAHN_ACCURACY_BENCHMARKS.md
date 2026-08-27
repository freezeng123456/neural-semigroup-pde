# Allen–Cahn forward-dynamics accuracy: literature sidecar

## Bottom line

There is no defensible global Allen–Cahn operator-learning SOTA. The literature has no standardized, directly comparable leaderboard for the exact task here: periodic 1D Allen–Cahn, `N=128`, `epsilon=0.1`, low-frequency Fourier initial conditions, an independent locked 500-trajectory test set, horizons `1.2/2.4/4.8`, and pointwise-grid MSE. Published results use different Allen–Cahn scalings, domains, boundary conditions, initial data, resolutions, rollout protocols, and usually relative `L2` rather than MSE.

The strongest directly relevant published number I found is **0.22% ± 0.05% relative `L2` error** for a 2D Allen–Cahn benchmark from the multi-physics m-PhOeNIX study; its single-physics FNO number is **0.22% ± 0.05%** as well (the table reports the same rounded value), while its 1D Allen–Cahn FNO result is **0.28% ± 0.17%**. Those are useful scale references, not a claim about our task. A separate long-time 1D Allen–Cahn study reports a time-averaged relative prediction error of about **`5.8e-4`** for transfer-learning DeepONet under one parameter setting and 10,000 training samples, but over `[0,50]` and with a different protocol. A classical PINN on a single fixed 1D initial condition reports mean `L2` solution error **`7.9e-3`**. None can be converted honestly into our locked-test MSE without rerunning the benchmark.

## What was checked

The table below records credible primary-source Allen–Cahn forward-dynamics benchmarks or official benchmark suites that are commonly used as comparators. “Not reported” means the cited source does not provide the requested detail in the benchmark result, not that the experiment did not have one.

| Primary source / model | Equation, domain, BC, parameters | Grid and initial data | Data, horizon, metric, reported accuracy | Directly comparable to this project? |
|---|---|---|---|---|
| **Lu et al., “A comprehensive and fair comparison of two neural operators” (CMAME 2022), official arXiv and code**: [paper](https://arxiv.org/abs/2111.05512), [repository](https://github.com/lu-group/deeponet-fno) | 1D Allen–Cahn benchmark; the paper’s benchmark configuration is not the project’s periodic `epsilon=0.1` setting. The cited result is presented as a relative `L2` operator benchmark; exact equation constants/BC should be taken from the released data/configuration rather than inferred from the table. | 1D grid and IC ensemble are source-specific; not the low-frequency Fourier/500-trajectory split. | Test relative `L2` error: **FNO `0.28% ± 0.17%`**, DeepONet **`1.36% ± 0.56%`**; the study also reports WNO `0.66% ± 0.03%`, CNO `1.36% ± 0.63%`, and a multi-physics model `0.22% ± 0.06%` for 1D Allen–Cahn. The table is not an MSE table and does not use the project’s three horizons. | **No.** Same broad PDE family and 1D operator-learning purpose; wrong/unmatched protocol and metric. |
| **m-PhOeNIX / “Multi-physics operator learning” (ICLR 2025 submission), official OpenReview PDF**: [PDF](https://openreview.net/pdf?id=ubUTIlAH0m) | 1D and 2D Allen–Cahn tasks in a multi-PDE benchmark. Exact coefficients, domain/BC, and discretization are defined in the paper’s benchmark section/configuration; they are not the project’s stated periodic `N=128, epsilon=0.1` specification. | Separate 1D/2D data-generation protocols; not the project’s Fourier-IC distribution and locked 500-trajectory test. | Relative `L2` error in percent. 1D: FNO **`0.28 ± 0.17%`**, DeepONet **`1.36 ± 0.56%`**, WNO **`0.66 ± 0.03%`**, CNO **`1.36 ± 0.63%`**, m-PhOeNIX **`0.22 ± 0.06%`**. 2D: FNO **`0.22 ± 0.05%`**, DeepONet **`0.83 ± 0.17%`**, m-PhOeN **`0.19 ± 0.04%`**. | **No.** Best useful published scale reference, but not an apples-to-apples MSE comparison. |
| **WNO/CNO/FNO/DeepONet comparison within the same m-PhOeNIX benchmark** | Same task definitions as the preceding row; these are separate baseline models, not independent standardized datasets. | Same source-specific 1D/2D grids and IC distributions. | Values above are the reportable numerical baseline range: approximately **`0.19–1.36%` relative `L2`** across the 1D/2D Allen–Cahn tables. | **No.** It is useful for model-family context, but counting these as separate external benchmarks would overstate the amount of independent evidence. |
| **Wang et al., “Transfer Learning Enhanced DeepONet for Long-Time Prediction of Evolution Equations”**, official arXiv: [arXiv:2212.04663](https://arxiv.org/abs/2212.04663) | 1D Allen–Cahn evolution equation; experiments vary the diffusion coefficient (`d1=1e-3, 5e-4, 1e-4`) and use the paper’s own spatial/temporal setup and BCs. | 1D discretization and IC generation are paper-specific; not the project’s low-frequency Fourier ensemble. | Time-averaged relative prediction error over **`[0,50]`**. For 10,000 training samples, TL-DeepONet reports **`5.81e-4`**, **`7.94e-4`**, and **`1.16e-2`** for the three listed diffusion settings; ordinary DeepONet is roughly order-one in the displayed table. | **No.** It is a valuable long-horizon 1D reference, but its horizon, diffusion parameters, data volume, and metric differ radically. |
| **Monaco & Apiletti PINN Allen–Cahn case**, as reproduced and numerically audited by Hillebrecht et al., “Efficient Error Certification for PINNs” (official arXiv): [paper](https://arxiv.org/abs/2305.10157) | `u_t + ρu(u²−1) − νu_xx = 0`, `t∈[0,1]`, `x∈[-1,1]`, `ρ=5`, `ν=1e-4`; fixed IC `u(0,x)=x² cos(πx)`; periodic value BC `u(t,-1)=u(t,1)`. | Continuous PINN evaluation rather than an operator dataset/grid; 6 hidden layers, 40 neurons/layer in the audited model. | Mean `L2` solution error **`7.9e-3`** (source’s description of the trained PINN). The certification table separately reports initial-condition empirical max squared error `1.60e-3`, periodic-boundary max squared error `5.66e-6`, and residual max squared error `10.74`; these are not interchangeable with solution MSE. | **No.** Periodic 1D is conceptually related, but it has one fixed IC, different coefficients, continuous PINN training, and no 500-trajectory forward-operator test. |
| **PDEBench, official benchmark suite and paper**: [NeurIPS/arXiv paper](https://arxiv.org/abs/2210.07182), [official repository](https://github.com/pdebench/PDEBench) | PDEBench standardizes several time-dependent PDE datasets and FNO/U-Net/PINN baselines, but the released core suite does **not provide an Allen–Cahn forward-dynamics leaderboard matching this task**. Its reaction-diffusion coverage should not be silently relabeled as Allen–Cahn. | Official datasets use equation-specific grids, parameter ranges, IC/BC distributions, and train/validation/test splits. | The suite reports model-specific metrics for its included equations, but no official Allen–Cahn number that can be quoted as the project’s comparator. | **No.** It is evidence that a benchmark API exists, not evidence of a directly comparable Allen–Cahn SOTA. |
| **Raissi et al., PINNs**, official paper/repository lineage: [paper](https://arxiv.org/abs/1711.10561), [DeepXDE repository](https://github.com/lululxvi/deepxde) | The canonical Allen–Cahn PINN example uses a bounded 1D space-time problem with periodic boundary constraints and a prescribed initial profile; it is a single-solution PINN rather than a learned map from an IC distribution. | Collocation points, not a fixed `N=128` operator grid; no independent trajectory ensemble. | The canonical source emphasizes solution plots and residual/constraint training rather than a standardized forward-operator MSE table. It should not be treated as a numerical leaderboard entry. | **No.** Historically important PINN reference, not a comparable multi-trajectory surrogate benchmark. |

### Interpretation of the numbers

Relative `L2` error and MSE answer different questions. If a paper reports

`relL2 = ||u_hat-u||_2 / ||u||_2`,

then neither the numerator nor the denominator is available from the scalar percentage alone; it cannot be converted into an absolute grid MSE. In addition, a time-averaged error over `[0,50]` cannot be compared directly with errors at the three project horizons. The PINN residual, initial-condition error, and boundary error in the certification paper are also constraint diagnostics, not prediction accuracy for a held-out initial-condition ensemble.

## What is the current best attainable/claimed accuracy?

The careful answer is two-part:

1. **Claimed published best within a named Allen–Cahn benchmark:** roughly **`0.2%` relative `L2`** is the lowest rounded number in the sources located here (m-PhOeNIX/multi-physics operator learning: `0.19–0.22%` for its reported Allen–Cahn settings). The strongest single-physics 1D FNO reference is about **`0.28% ± 0.17%` relative `L2`**.
2. **Best long-time 1D number located:** about **`5.8e-4` time-averaged relative prediction error** for TL-DeepONet in one diffusion setting over `[0,50]`. This is numerically smaller, but it is not evidence of a lower absolute error on the project’s task because the denominator, time averaging, parameter regime, data distribution, and test protocol differ.

Therefore, if the physics-anchored result were the only contribution, the paper should describe its absolute MSE as a **new, tightly specified benchmark result**, and compare it to these numbers only as contextual ranges. It should not call the result global SOTA on the basis of the current literature.

## Recommended reporting for this paper

Report, for each horizon `T∈{1.2,2.4,4.8}`:

- mean and standard deviation of the per-trajectory grid MSE over the locked 500-trajectory test set, with a 95% confidence interval;
- normalized MSE or relative `L2` **in addition to**, never instead of, absolute MSE, with the exact denominator and spatial/time averaging convention;
- median, 90th/95th percentile, and worst-trajectory MSE, because Allen–Cahn transitions can make the mean hide failures;
- the grid, de-aliasing/reference solver, time step, epsilon/coefficient convention, periodic implementation, IC Fourier-mode distribution, training-set size, parameter count, rollout/integration work, and random seeds;
- error versus physical horizon, not only the final horizon, plus energy/phase diagnostics if the physics anchor is part of the claim.

The headline should be phrased like: “On our periodic 1D `N=128`, `epsilon=0.1` low-frequency-Fourier benchmark, the physics-anchored model achieves absolute test MSE … at horizons … under a locked 500-trajectory evaluation.” Add the published relative-`L2` values in a separate context table and explicitly mark them “not directly comparable.”

## What is required for an SOTA claim?

A credible claim would require a new standardized experiment, not a literature-number conversion. Freeze and release:

1. the exact PDE and coefficient/epsilon convention, periodic domain, grid, reference solver and tolerances;
2. a deterministic IC generator (including Fourier cutoff and coefficient law), public train/validation/test trajectory IDs, and the locked 500-trajectory test set;
3. fixed horizons `1.2/2.4/4.8`, absolute grid MSE as the primary metric, and a secondary relative `L2` metric;
4. matched implementations and budgets for at least FNO, DeepONet, PINO, a discrete autoregressive neural PDE solver/message-passing model, and a PINN or PDE-Net-style baseline where the latter is meaningful for the operator task;
5. matched parameter count, optimizer/data budget, rollout steps or vector-field evaluations, and at least three seeds, with code/checkpoints/configurations released;
6. a clear separation between interpolation and extrapolation in time, one-shot/direct prediction and autoregressive rollout, and physics information available to every baseline in the relevant ablation;
7. uncertainty intervals and a statistical test or paired bootstrap on the same 500 trajectories.

Only after those baselines are run on the same released split should the paper use “SOTA,” and then only as **SOTA on the newly defined standardized benchmark**. A stronger and safer initial claim is “best reported accuracy on our benchmark” or “a new reproducible periodic 1D Allen–Cahn benchmark,” unless the standardized baseline study actually establishes a win.

## Primary sources

- Lu et al., *A comprehensive and fair comparison of two neural operators with practical extensions*, CMAME 2022. [arXiv](https://arxiv.org/abs/2111.05512), [official code](https://github.com/lu-group/deeponet-fno)
- Multi-physics operator-learning study, official OpenReview PDF. [PDF](https://openreview.net/pdf?id=ubUTIlAH0m)
- *Transfer Learning Enhanced DeepONet for Long-Time Prediction of Evolution Equations*. [arXiv](https://arxiv.org/abs/2212.04663)
- Hillebrecht et al., *Efficient Error Certification for Physics-Informed Neural Networks*. [arXiv](https://arxiv.org/abs/2305.10157)
- Takamoto et al. et al., *PDEBench: An Extensive Benchmark for Scientific Machine Learning*. [arXiv](https://arxiv.org/abs/2210.07182), [official repository](https://github.com/pdebench/PDEBench)
- Raissi et al., *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. [arXiv](https://arxiv.org/abs/1711.10561)
