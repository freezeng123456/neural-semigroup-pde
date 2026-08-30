# Fisher--KPP generator and stability diagnostic protocol

Status: frozen exploratory protocol before observing any result from this
diagnostic matrix.

Date frozen: 2026-08-31

## 1. Question and evidence boundary

The locked Fisher experiment showed that Model A was substantially more
composition-consistent than Model B, while its rollout MSE was essentially
tied and seed dependent.  The next question is therefore not whether the
composition result repeats.  It is whether the additional terms required by
the transfer theorem—generator matching, one-sided stability, and learned-flow
time-discretization error—explain why composition consistency alone did not
produce a material accuracy advantage.

Every run in this protocol is `exploratory=true` and
`do_not_use_for_formal=true`.  The evaluator is checkpoint-only.  It performs
no training, fine-tuning, optimization, checkpoint selection, cache creation,
or cache modification.  The six outputs use new canonical roots and cannot
change the Section 7.2 Fisher decision.

## 2. Frozen inputs and matrix

The matrix has six independent single-GPU evaluator cells
(three seeds times two models).  Each cell evaluates both frozen query lags:

- seeds: `31415`, `271828`, `161803`;
- models: A (`latent`) and B (`latent_query_time`);
- query lags: `0.075`, `0.15`;
- cache: all 500 samples in the immutable formal Fisher locked cache;
- reference snapshot times: `0`, `0.6`, `1.2`, `2.4`, `4.8`;
- learned-path subset: the first 64 locked samples;
- refinement subset: the first 32 locked samples;
- refinement horizon: `1.2`;
- RK4 substeps per repeated lag call: `8`, `16`, `30`, `60`, with `120`
  substeps per call used only as the frozen finest surrogate.

Frozen provenance:

- source commit: `637345584dc2db8ddccf9116a995615c3c036104`;
- source archive SHA-256:
  `a370efa4af9bbb11fbcd72ef422410f651f8f2ad28eee561c45e766c88fcadf5`;
- locked cache SHA-256:
  `29d0e4b36e9d758f87264ae1d555d865c9e037776903e1cab448a0a8782b490e`;
- the six checkpoint SHA-256 values are those already recorded in
  `SEMIGROUP_NOVELTY_EXPERIMENT_RESULTS.md` and must be supplied explicitly to
  each launcher.

Every input is hashed before and after evaluation.  A mismatch is a failed
run.  No output root may be reused after success or failure.

## 3. Physical-coordinate generator diagnostic

The periodic spectral reference generator is
\[
A_h(u)=\nu D_{xx,h}^{\mathrm{spec}}u+r u\odot(1-u).
\]
For \(z=E(u)=\operatorname{logit}(\operatorname{clamp}_\epsilon u)\) and
represented state \(v=D(z)=\sigma(z)\), the learned physical-coordinate
generator is
\[
F_{\theta,h}^{u,[\tau]}(v)
=v\odot(1-v)\odot f_{\theta,h}^{[\tau]}(z).
\]
For A the right-hand side is independent of \(\tau\); for B it is evaluated
separately at the two frozen query lags.

At every reference snapshot, and at learned-flow snapshots generated under the
same frozen query lag, first form \(v=D(E(u))\).  Report the representation
error \(\lVert v-u\rVert_h\), raw and represented state ranges, and the
per-represented-state mesh-\(L^2\) residual
\[
r_{\mathrm{gen}}(v)=
\lVert F_{\theta,h}^{u,[\tau]}(v)-A_h(v)\rVert_h,
\]
its relative version with denominator
\(\max(\lVert A_h(v)\rVert_h,10^{-12})\), and mean, median, P90, P95, maximum,
RMS numerator, RMS denominator, and finite counts.  Also report cosine
alignment between learned and reference generators when both norms are above
the denominator floor.  If no state clears that floor, cosine alignment is
explicitly unavailable rather than an evaluator failure.

The sample maximum is a diagnostic on the recorded finite represented-state
set, not a certified uniform generator-error bound or a residual at the raw
cache state when representation error is nonzero.  The common observed set is
the union of represented reference and learned-path states; it is not asserted
to be invariant or to equal the transfer theorem's common state set.

## 4. One-sided stability diagnostic

For every reference snapshot, use exactly the same frozen raw-state pair
construction for all six checkpoints.  Each checkpoint evaluates the
corresponding represented pair after its own encoder/decoder map:

1. cyclic cross-sample pairs \(v_i,w_i=v_{i+1}\);
2. deterministic sinusoidal perturbations of mesh-RMS amplitude `0.001`;
3. deterministic sinusoidal perturbations of mesh-RMS amplitude `0.01`.

Only the sinusoidal perturbation pairs are clipped to
`[1e-5, 1-1e-5]`; cyclic pairs reuse the cache states without an additional
raw-state clip.  The represented pair range and actual nonzero pair distance
are recorded.  For the reference and learned generators report
\[
q_F(v,w)=
\frac{\langle F(v)-F(w),v-w\rangle_h}
{\lVert v-w\rVert_h^2}.
\]
The evaluator stores the same distribution summaries and the finite-sample
maximum.  The reference implementation sanity gate is
\[
\max q_{A_h}\leq r+10^{-4}.
\]
This follows algebraically for the spectral Laplacian on states in
\([0,1]^N\); failure indicates an implementation, state-range, or metric bug.
No analogous pass threshold is imposed on the learned models, and their sample
maxima are not called global one-sided Lipschitz constants.

## 5. RK4 refinement diagnostic

For each lag, advance the same first 32 initial conditions to \(H=1.2\) by
repeated lag calls.  Each call uses `8`, `16`, `30`, `60`, or `120` RK4
substeps.  The `120`-substep result is the frozen finest surrogate; it is not
an exact solution.  Every row records the composition depth, per-call and
total RHS counts, saturation fraction, and the error relative to the finest
surrogate.

For each non-finest level report
\[
E_m=\lVert u_m(1.2)-u_{120}(1.2)\rVert_h
\]
and its relative version.  This is a learned-flow numerical difference from a
finest surrogate, not error against the exact learned flow.  Report pairwise
observed orders where both adjacent differences exceed `1e-12`; do not force
an order estimate at a floating-point floor.  At the production value `m=30`,
also report the confounding diagnostic
\[
R_{\mathrm{num/cache}}
=\frac{\operatorname{RMS}\lVert u_{30}-u_{120}\rVert_h}
{\operatorname{RMS}\lVert u_{30}-u_{\mathrm{cache}}\rVert_h}.
\]
Here \(u_{\mathrm{cache}}\) is the locked cache reference, itself a spatial and
temporal numerical surrogate.  The denominator therefore mixes learned-model
approximation, representation/re-encoding, and reference-discretization
effects; the ratio is not a pure decomposition of PDE error.  Learned-flow
numerical error is labelled materially confounding when this ratio exceeds
`0.10`.  This diagnostic does not re-evaluate the formal MSE decision.

Every repeated lag call contributes to aggregate diagnostics of whether
\(E(D(z))=z\) holds numerically: the number of decoded coordinates at or beyond
the encoder clamp threshold, and the maximum/norm of the re-encoding
discrepancy.  A clean decoded-semigroup interpretation requires zero
encoder-clamp activations as well as zero
\(\lvert z\rvert\geq19.999\) latent-clamp events.

## 6. Frozen interpretation branches

The aggregate reports the following branches without changing any setting:

1. **Generator-matching branch.**  A supports a material generator-residual
   advantage only if its cross-seed geometric-mean RMS relative residual is at
   most `0.90` times B for both query lags on the reference snapshots.
2. **Empirical-stability branch.**  A supports a stability advantage only if
   its per-seed maximum learned one-sided quotient is no larger than B's in at
   least two of three seeds for both query lags.  This is an empirical ordering,
   not a theorem.
3. **Learned-flow numerical branch.**  Production RK4 is called non-dominant only
   if `R_num/cache <= 0.10` in every seed/model/lag cell.  Otherwise the failing
   cells are listed and no structural explanation may ignore integrator error.
4. **Explanation branch.**  If A has no material generator-residual advantage
   or no empirical-stability advantage while learned-flow numerics are
   non-dominant, the tied formal MSE is consistent with the layered theorem:
   autonomy reduced numerical composition defect without reducing all terms
   controlling PDE error.  If A improves both missing terms yet MSE remains
   tied, the next obligation is spatial/reference consistency rather than more
   composition tests.

These branches are deliberately asymmetric with the already observed formal
result: they diagnose that result but cannot replace or amend it.

## 7. Engineering completion gate

A cell is complete only after normal process exit and all of the following:

- parseable strict JSON and CSV outputs;
- all requested snapshot, lag, pair-family, and refinement rows;
- finite metric values except explicitly unavailable observed orders and cosine
  alignment;
- the reference one-sided sanity gate;
- A's physical generator agrees across the two nominal query-lag labels to
  within `1e-7` on the same states;
- no detected latent saturation at `|z| >= 19.999`;
- no encoder-clamp activation during decoded/re-encoded repeated lag calls;
- identical checkpoint, cache, and source-archive hashes before and after;
- an immutable manifest, result, receipt, and literal `passed` marker;
- independent six-cell aggregation from the receipts rather than copied log
  text.

A smoke may reduce sample counts and refinement levels, but it validates only
execution and artifact shape.  It cannot support a scientific branch.
