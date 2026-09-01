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
loss and architecture.  The completed distribution diagnostic shows that the
epoch-100 checkpoint improves generator residual by about 2% on training and
initial states, but the sign reverses after learned rollout.  Rollout-selected
best checkpoints reduce the later deterioration while losing the
initial-state improvement.  The failure is trajectory-distribution transfer,
not normalization.  The next step is a trajectory-tube error bound and a
matching trajectory-distributed intervention, not a loss-weight sweep or
another seed campaign.

Machine-readable values and execution evidence are stored under
`results/20260901-fisher-generator-consistency-8fdbae8-3seed-r1/`.
The follow-up localization is stored under
`results/20260901-fisher-generator-distribution-72505be-r1/`.
