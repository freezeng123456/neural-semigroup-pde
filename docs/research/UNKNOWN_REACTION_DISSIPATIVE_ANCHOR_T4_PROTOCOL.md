# Known dissipation / unknown source: post-screen structural ablation

Frozen before training, 2026-09-07. This is a NEW exploratory follow-up to
`UNKNOWN_REACTION_MATCHED_T4_PROTOCOL.md`, not an independent confirmation.

## Trigger and hypothesis

The first 24-cell screen completed: A/B pooled MSE=1.00727; neither learned
model is useful in absolute accuracy (primary MSE about 1, versus pure heat
0.318). All 24 selected reaction curves point outward throughout the sampled
tail |u|>=1.05, where the reference reaction is restorative. One-step
validation misses this long-trajectory failure. These findings do not identify
autonomy as its cause and do not justify width/weight/epoch searches.

One minimal representation intervention is now frozen: give BOTH models the
same known cubic dissipation and learn only the unknown source:
`r_theta(u,tau) = -u**3 + neural_source(u,tau)`.
The physical problem and targets remain exactly the same. The unknown source
is still learned ONLY from input/output state pairs; the learner is never
given its formula `u+0.2*sin(3*u)` or its derivatives.
This changes the available prior information. It does NOT claim successful
identification of an entirely unknown reaction under the original setting.
The neural source is bounded as a function of u for fixed weights, making
the cubic dominate at sufficiently large amplitudes for the continuous
reaction field. This does not assert finite-step RK/Euler invariance or an
invariant interval with the particular radius 1.2.

## Fixed comparison

Reuse the exact first-screen cache and nested 16/128 training subsets, three
seeds, paired initial weights, 65 parameters, learning rate 0.01, 120 updates,
validation selection, diffusion heat map, substep budgets 1 and 4, and all
evaluation lags/horizons: another 24 cells. No hyperparameter changes.
Report every failed or nonfinite cell; do not rerun until successful.

Primary structural-ablation endpoint: pooled MSE(new)/MSE(original) on the
same unseen-lag/horizon primary cells, separately for A and B. The known
dissipation hypothesis requires both pooled ratios <=0.90, at least 2/3 seeds
per model meeting <=0.90, and no increase in pooled bound-violation fraction.

The autonomous A/B accuracy and refinement rules remain EXACTLY those in the
parent protocol, and their pass/fail is reported separately. If both A and B
improve but A/B does not materially favor A, attribute evidence to the shared
dissipative prior, not to temporal autonomy.

Retain pure heat and oracle reaction baselines and add the known cubic alone
(no learned source) at both budgets. Report absolute errors, bound violations,
physical energy, scalar generator curves and refined-checkpoint error. The
same test cache is deliberately reused for paired ablation; it is no longer
an untouched confirmation set. No results from this phase alter the formal
Fisher lane or the first phase's frozen decision.

After this one ablation, stop further training in this task, report both
phases and all failed claims, and recommend the next research decision from
the full evidence rather than escalating compute.
