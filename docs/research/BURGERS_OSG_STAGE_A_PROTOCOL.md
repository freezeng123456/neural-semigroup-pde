# Burgers OSG stage A: literature-informed exploratory comparison

Frozen on 2026-10-04 before new model training. This is a bounded exploratory
study, not a new blinded confirmation. The reference-v2 test results have
already been inspected in the preceding study. No choices below depend on
stage-A test results.

## Question and scope

Does in-range soft semigroup training improve a direct OSG map relative to
the identical map without that training? Retain the existing small autonomous
and duration-conditioned generators as context. Cross-architecture differences
do not identify the causal effect of hard versus soft semigroup structure.

The user authorized Beijing Super Cloud Computing Center on 2026-10-04.
Use one allocated V100, a <=15-minute preflight, then one <=2-hour sequential
study job. Run three seeds with six trainable cells each. Do not extend the
budget, add noise/history/all2all, or tune the matrix after seeing test results.
If engineering/runtime failures occur, preserve their artifacts and repair
implementation errors before a clearly versioned retry.

## Provenance and reference

Reuse the independently audited reference-v2 cache, SHA-256
`4b96d9397a46c22cbf2d71b92706b756e720a2cfa08fac46f6f823e5931ec9a0`.
All cache contract checks from run_burgers_validated.py must pass.
Data: 1000 train, 50 validation, 128 test; N512 float64 dealiased reference,
low-pass to N64 with omitted coarse Nyquist, nu=.01 and L=2*pi.
Train lags .025/.05/.075/.1; test lags .04/.08, rollout horizons .4/.8.
One assigned training lag per initial condition, exactly as reference-v2.
No additional supervised snapshots are used.

Freeze a Git commit, source archive hash, resolved settings, author-code
revision, cache hash and unique remote output directory before submission.
Keep historical runners/results unchanged. Existing Python environment may
be reused read-only; record actual Python, Torch, CUDA and GPU versions.

## Model matrix

Seeds: 31415, 271828, 161803. Six cells per seed:

| Label | Model | Nominal parameters | Semigroup weight |
| --- | --- | ---: | ---: |
| small_autonomous | historical r0/h8/l1 conservative flux + fixed diffusion | 33 | hard autonomous continuous flow |
| small_query | identical generator with requested duration input | 33 | none |
| osg33_free | periodic r2/h4/l1 local OSG residual map | 33 | 0 |
| osg33_gdsg | same OSG map | 33 | 1 |
| osg1313_free | periodic r2/h32/l2 local OSG residual map | 1313 | 0 |
| osg1313_gdsg | same OSG map | 1313 | 1 |

The two OSG capacity levels avoid drawing conclusions from only a very narrow
backbone. The primary comparisons pair lambda within each OSG capacity.
They share exact initial tensors, supervised minibatches and optimizer rules.
The historical A/Q pair also shares initial tensors. Nominal counts do not
imply equal function classes: the flow's final constant flux bias cancels in
the divergence, and A has eight inactive control weights. Identified functional
parameter upper bounds are 24 and 32; this is not an exact intrinsic dimension.
OSG does not inherit the conservative divergence or the known diffusion anchor.

## OSG adaptation and author implementation check

Source: Deep-OSG (JCP 2023) and DUE, author repository
https://github.com/AI4Equations/due, revision
`1e629104a12a9c3e3064996cbd4e301cec641ad8` (LGPL-2.1).
Our implementation follows the mathematical formulas independently; it is a
periodic local shared-MLP adaptation, not a reproduction of the paper's FNO
architecture or its published benchmark numbers. A separate scalar decay
software check exercises the pinned author's OSG implementation.

The map is u + tau * scale * N((patch(u)-center)/scale, normalized_tau).
Center and scale use the global minimum/maximum of training u0 only, stored
as buffers. No per-trajectory normalization or clipping. Normalize the time
feature using the training lag endpoints, but multiply by physical tau and
add physical s+t before time normalization. Losses are physical-unit MSE.
The old generator definition is unchanged; comparisons across these input
parameterizations remain descriptive.

Use GDSG = .5*(MSE(Phi_t(Phi_s(u)),Phi_(s+t)(u)) +
MSE(Phi_s(Phi_t(u)),Phi_(s+t)(u))). Keep gradients through both compositions
and the direct branch. The objective is (fit + lambda*GDSG)/(1+lambda), as in
DUE. At lambda=1 this also rescales the supervised gradient; interpret the
result as the normalized combination-training intervention, not as an
isolated regularizer-strength effect. Lambda=0 skips all extra calls.

One collocation state per supervised sample (Q=1), drawn from training u0.
Independently sample total duration uniformly on [.05,.1], then split it
uniformly with both parts >=.025. Thus s, t and s+t remain in training support.
This deliberately changes the author's box-state sampling and independently
sampled times (whose sum can exceed the supervised lag range). Do not call
this adaptation an exact replication of the original GDSG protocol.
Use a separate fixed RNG stream so collocation cannot change minibatch order.

## Training, selection, checkpointing

100 epochs, batch64, AdamW lr=.001, weight decay=1e-5, cosine schedule,
gradient norm clip1, float32; flow training uses12 RK4 substeps.
Validate every10 epochs and at the final epoch, by MSE over the four training
lags on all50 validation trajectories. Earliest epoch wins ties.
Fix all six selected checkpoints for a seed before its test evaluation.
This remains a fixed-supervised-exposure comparison, not equal compute or
proof of convergence. Per supervised sample: OSG-free one network call,
GDSG six, flow48 flux-network calls. Record calls, wall time and GPU memory.

Save selected and final weights, optimizer, scheduler, epoch/update count,
RNG streams, initial/final digests, validation history and batch-order digest.
Explicit pilot interruption/resume is limited to scheduled validation boundaries,
so an interruption cannot introduce an extra checkpoint-selection candidate.
The preflight validates planned short-run save/resume. The full study runner
does not automatically recover an interrupted multi-model job; a full retry must
use a new output root and preserve the failed run. No unattended retry is configured.
Preflight must verify CPU/GPU prediction agreement, finite backward passes,
parameters changing, deterministic repeated short runs, and exact continuation
of a save/resume on the same device. Never equate checkpoint presence with
successful training or resume.

## Evaluation and decision

Retain per-trajectory MSE/relative L2, spectral error, mean drift and resolved
energy diagnostics at rollout steps. Energy monotonicity is a diagnostic;
projected fine trajectories need not have monotone resolved energy.
Include persistence, zero prediction and the exact-flux centered-difference
baseline. The zero predictor is a reference prediction, not a flow satisfying
the t=0 identity.

Flow evaluation uses12 substeps at .04 and24 at .08, matching total RK work.
OSG uses one direct-map call per rollout step. Report actual costs separately.
Report absolute in-range composition defects and cross-lag endpoint differences;
do not divide by the autonomous numerical-sequence zero. No out-of-support
direct .4/.8 map is used to rank OSG models.

For each OSG capacity, primary statistic is the geometric mean of GDSG/free
MSE ratios over four lag/horizon conditions and three seeds. Report all three
paired seed ratios. A material exploratory result requires ratio<=.90 and at
least2/3 seeds favoring GDSG. This is not a statistical significance test.
The four conditions share initial states, so they are not independent repeats.
Report both capacity comparisons regardless of outcome; no best-capacity
selection on test. A/Q is secondary; cross-architecture ranking descriptive.

Nonfinite training stops the job and preserves evidence. Nonfinite rollout
is a reported model failure, never silently capped or omitted. Hash cache and
sources again after the run; aggregate only complete matching source/config
receipts. No test-driven additional training is part of this protocol.
