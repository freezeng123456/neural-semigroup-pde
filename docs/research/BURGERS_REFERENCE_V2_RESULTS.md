# Burgers reference-v2 results

Completed locally on 2026-10-03. Evidence class: exploratory.

The validated-reference study completed all 12 models: two architectures,
two temporal rules, and three training seeds. The autonomous model did not
meet the material accuracy threshold for either architecture. The small
generator reduced error substantially at the fixed 100-epoch budget.

## Primary comparisons

| Comparison | Geometric-mean MSE ratio | Seeds favoring numerator |
| --- | ---: | ---: |
| Small autonomous / small query | 1.075021876 | 0/3 |
| Large autonomous / large query | 1.015429317 | 0/3 |
| Small / large, autonomous | 0.140817513 | 3/3 |
| Small / large, query | 0.133011462 | 3/3 |

The frozen material rule requires A/B <= 0.90 and at least two seeds favoring
A. Both architecture comparisons fail. These ratios combine three seeds and
four lag/horizon conditions; the conditions share 128 held-out initial states.
They are not independent repeated trials. All 12 checkpoints selected epoch
100, so this remains a fixed-budget result.

The small architecture uses radius 0, width 8, one hidden layer, and 33 nominal
parameters. The large architecture uses radius 2, width 32, two hidden layers,
and 1,313 parameters. Paired A/B models share initial tensors and minibatch
order. Autonomous models retain 8 or 32 inactive control-column weights;
effective functional capacity is therefore not matched.

## Physical and numerical diagnostics

| Model | Mean rollout MSE | Minimum monotone-energy fraction | Mean cross-lag RMS defect |
| --- | ---: | ---: | ---: |
| Small autonomous | 0.000407109 | 1.000000 | 0 |
| Small query | 0.000384234 | 1.000000 | 0.00469617 |
| Large autonomous | 0.00442522 | 0.406250 | 0 |
| Large query | 0.00422714 | 0.789063 | 0.00829349 |
| Exact-flux centered-difference baseline | 0.000432241 | 1.000000 | 0 |

Mean MSE uses an arithmetic average over seeds and conditions. The minimum
energy fraction is the worst condition/seed, using a 1e-8 increment tolerance.
All measured small-model rollouts passed; this is a sampled observation.
The large models had positive energy increments up to 0.005466 and 0.001236.
The target is a projected fine state, whose resolved energy need not always
decrease.

Autonomous cross-lag paths use identical RK4 step sequences. Their exact zero
defect here establishes consistency of that evaluation, without establishing
PDE accuracy. Query conditioning produces an in-range endpoint difference.
The matched-condition direct/composed control is near rounding error. Doubling
the number of substeps changes model endpoints by at most 5.13e-7 RMS, with
average squared differences about 4e-14. RK4 resolution is adequate relative
to the observed prediction errors. Maximum spatial-mean drift is below 8.54e-8.

## Reference validation and target scope

Production data use float64 dealiased Fourier Galerkin evolution at N=512,
dt=0.0005, viscosity 0.01, with low-pass projection to N=64 and omitted coarse
Nyquist mode. Separate seeds generate 1,000 training, 50 validation, and 128
test states. Training lags are .025/.05/.075/.1; evaluation lags are .04/.08
and horizons are .4/.8. Both evaluation lags use equal total RK4 work.

The independent 50-state reference audit passed its frozen thresholds:

- Maximum N512/N1024 relative L2 discrepancy: 0.000512509, below 0.001.
- Maximum discrepancy on halving dt: 6.24e-12, below 1e-5.
- Fine-grid energy and mean checks passed during audit and cache generation.
- Analytic Cole-Hopf and Galerkin energy-identity tests passed.

Repeated coarse projection/restart at intervals .04/.08 gives pooled relative
L2 differences of 2.466%/1.148% at horizon .8 on the audit states. This diagnoses
the specific restart procedure. A projected fine PDE is not assumed to be a
closed Markov flow on 64 values. Reference resolution and initial-condition
sampling both changed, so old/new ranking changes cannot be attributed solely
to the reference solver.

## Evidence and reproduction

- [Frozen protocol](BURGERS_REFERENCE_V2_PROTOCOL.md)
- [Compact study record](../../results/20261003-burgers-reference-v2/README.md)
- [Aggregate JSON](../../results/20261003-burgers-reference-v2/aggregate/aggregate.json)
- [60-row cell table](../../results/20261003-burgers-reference-v2/aggregate/cells.csv)
- [Reference audit](../../results/20261003-burgers-reference-v2/data/reference_audit.json)
- [Repeated-restart diagnostic](../../results/20261003-burgers-reference-v2/reference_review_v2.json)
- [Run card](../../results/20261003-burgers-reference-v2/run_card.json)

Training source/protocol commit: `4919e3726cb4edd10fa64a7962266974eea1db7f`.
Cache SHA-256:
`4b96d9397a46c22cbf2d71b92706b756e720a2cfa08fac46f6f823e5931ec9a0`.
All three processes exited with code 0. The aggregator verified completion,
configuration/cache/checkpoint hashes, paired initialization, and metric
consistency with per-sample records. The complete suite passed 273 tests in
38.28 seconds on CPU. Source, data, checkpoints, histories, diagnostics, and
logs are retained; no GPU parity claim follows from this CPU study.

## Research decision

Use the small architecture as the Burgers baseline. Preserve both temporal
rules and the fixed exact-flux comparator. The next accuracy study should
address the noise-dependent result on the later GitHub confirmation branch:
its six-seed A/B ratios are 1.337859 without noise and 0.332751 at noise .01.
Its effective parameter counts differ (97/129). A frozen noise-by-capacity
comparison is needed before attributing the benefit specifically to autonomy.

The new Fisher in-range evaluator is ready with 21 tests, but historical
checkpoint evaluation remains unrun because 12 exact inputs are missing.
See [requirements and usage](FISHER_INRANGE_ATTRIBUTION_USAGE.md).
