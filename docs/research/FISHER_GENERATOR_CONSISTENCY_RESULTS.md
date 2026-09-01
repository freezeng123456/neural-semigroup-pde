# Fisher--KPP generator-consistency result

The frozen three-seed screen in
`FISHER_GENERATOR_CONSISTENCY_SCREEN.md` is complete.  Direct physical
generator matching with weight `0.01` did **not** improve the independent
generator residual: the geometric-mean paired ratio was `1.0246`, versus the
pre-registered material threshold `0.90`.  The geometric-mean rollout-MSE
ratio was `1.0452`; only one of three seeds improved.

This is a useful negative result.  It separates three statements that should
not be conflated:

1. the autonomous architecture defines one continuous-time generator;
2. the sigmoid decoder enforces the state range by construction;
3. a raw pointwise generator-MSE penalty transfers to an independent,
   normalized generator residual and improves long rollouts.

The first two remain true.  This experiment rejects the third for the tested
loss and architecture.  The next step is an evaluation-only distribution and
scale decomposition, not a loss-weight sweep and not another seed campaign.

Machine-readable values and execution evidence are stored under
`results/20260901-fisher-generator-consistency-8fdbae8-3seed-r1/`.
