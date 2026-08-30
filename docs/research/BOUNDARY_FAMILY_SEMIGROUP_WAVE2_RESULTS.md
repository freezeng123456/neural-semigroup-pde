# Boundary-Family Neural Semigroup Wave 2: completed exploratory results

Date: 2026-08-30

## Outcome

The frozen 54-cell Wave 2 matrix completed successfully. All three spatial
boundary families passed the hard boundary implementation gate, and all three
passed the preregistered autonomous-versus-query-time temporal rule. The
result supports a clearer two-part research object:

\[
\underbrace{X_B=\{u:B_hu=g\}}_{\text{boundary-admissible state space}}
\quad+\quad
\underbrace{\dot u=F_\theta(u),\;F_\theta(u)\in T_uX_B}_{
  \text{one tangent autonomous generator}}
\quad\Longrightarrow\quad
\{S_t:X_B\to X_B\}_{t\geq0}.
\]

The boundary map supplies spatial admissibility. Autonomy supplies temporal
time homogeneity. The semigroup law alone does not create a Dirichlet,
Neumann, or Robin boundary condition.

This is exploratory mechanism evidence for one one-dimensional autonomous
reaction--diffusion PDE. It is not part of the locked Fisher--KPP formal lane
and is not a theorem about arbitrary PDEs, geometries, or discretizations.

## Frozen experiment

The reference equation was

\[
u_t=0.02u_{xx}+u-u^3,\qquad x\in(0,1),
\]

on a 33-point grid. The matrix crossed:

- boundary family: nonhomogeneous Dirichlet, homogeneous Neumann, and Robin;
- enforcement: hard, fixed penalty with `lambda=1`, and unconstrained;
- temporal rule: autonomous and query-time-conditioned;
- training seed: `31415`, `271828`, and `161803`.

Every full cell had 1,249 parameters. Within a seed, all 18 cells had the same
initial parameter fingerprint. Each family used one immutable cache, and all
cells used source commit `148f4946fa489ec2b0bf485cf76bc62c7456b3e3`.
Checkpoint selection used validation prediction MSE only. The complete frozen
design is in `BOUNDARY_FAMILY_SEMIGROUP_WAVE2_PROTOCOL.md`.

## Primary hard-boundary result

All values below are geometric means across the three paired seeds unless a
maximum is stated. `Defect ratio` and `MSE ratio` are autonomous divided by
query-time, so a value below one favors the autonomous model.

| Boundary | Maximum hard-boundary evidence | Autonomous nonuniform defect | Query-time nonuniform defect | Defect ratio | Autonomous long MSE | Query-time long MSE | MSE ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Homogeneous Neumann | `0.0` | `3.14297e-7` | `7.94439e-3` | `3.95622e-5` | `1.44307e-5` | `1.15483e-4` | `0.124960` |
| Nonhomogeneous Dirichlet | `0.0` | `4.62662e-7` | `8.36985e-3` | `5.52772e-5` | `1.29954e-5` | `1.18248e-4` | `0.109899` |
| Robin | `1.02445e-6` | `2.80305e-7` | `8.94085e-3` | `3.13511e-5` | `1.45332e-5` | `1.64057e-4` | `0.0885864` |

The hard boundary threshold was `5e-6`. Dirichlet and Neumann reached exact
float32 zero in the recorded evidence; Robin remained below the threshold.
All hard boundary-stress final residuals were zero after deliberately
perturbing the endpoints by `0.25`.

For every family, the autonomous defect and autonomous long-rollout MSE were
lower in all `3/3` paired seeds. The defect reductions were approximately
25,277-fold for Neumann, 18,091-fold for Dirichlet, and 31,897-fold for Robin.
The descriptive long-rollout MSE reductions were approximately 8.00-fold,
9.10-fold, and 11.29-fold, respectively.

## The two axes do different work

The autonomous cells below isolate the enforcement comparison. `Stress max`
is the largest final boundary-stress residual across the three seeds.

| Boundary | Enforcement | Nonuniform defect GM | Long-rollout MSE GM | Stress max |
| --- | --- | ---: | ---: | ---: |
| Homogeneous Neumann | hard | `3.14297e-7` | `1.44307e-5` | `0.0` |
| Homogeneous Neumann | penalty | `1.45813e-7` | `1.29069e-2` | `0.287890` |
| Homogeneous Neumann | unconstrained | `2.53226e-7` | `5.48148e-3` | `0.202546` |
| Nonhomogeneous Dirichlet | hard | `4.62662e-7` | `1.29954e-5` | `0.0` |
| Nonhomogeneous Dirichlet | penalty | `1.47435e-7` | `1.15271e-2` | `0.257448` |
| Nonhomogeneous Dirichlet | unconstrained | `7.20647e-7` | `5.25212e-3` | `0.292364` |
| Robin | hard | `2.80305e-7` | `1.45332e-5` | `0.0` |
| Robin | penalty | `1.30727e-7` | `8.79598e-3` | `0.194589` |
| Robin | unconstrained | `2.28161e-7` | `1.16360e-3` | `0.108766` |

This is the most useful negative control in Wave 2. Penalty and unconstrained
autonomous models can have a composition defect of order `1e-7` while badly
violating the spatial boundary and producing much larger long-rollout error.
Therefore:

- low temporal composition defect is not sufficient for boundary
  compatibility;
- low temporal composition defect is not sufficient for PDE accuracy;
- the fixed `lambda=1` final-output penalty did not provide a reliable
  boundary guarantee and sometimes had a larger stress residual than the
  unconstrained model;
- hard boundary construction and autonomous time construction are separate,
  complementary interventions.

Autonomy reduced the nonuniform defect in all `3/3` seeds within every one of
the nine boundary-family/enforcement combinations. It also reduced the
descriptive long-rollout MSE in all those paired comparisons, but the MSE
reduction was only about 5--17% for the penalty/unconstrained models and about
8--11 fold for the hard models. This interaction is descriptive; Wave 2 did
not preregister a factorial interaction test.

## Refinement distinguishes structure from integration error

The primary statistic used eight RK4 steps per nonuniform partition segment.
The frozen refinement diagnostic was:

| Boundary | Temporal rule | 4 steps | 8 steps | 16 steps |
| --- | --- | ---: | ---: | ---: |
| Homogeneous Neumann | autonomous | `1.04950e-5` | `3.14297e-7` | `1.12570e-7` |
| Homogeneous Neumann | query-time | `7.94561e-3` | `7.94439e-3` | `7.94441e-3` |
| Nonhomogeneous Dirichlet | autonomous | `2.53225e-5` | `4.62662e-7` | `1.15823e-7` |
| Nonhomogeneous Dirichlet | query-time | `8.41355e-3` | `8.36985e-3` | `8.36987e-3` |
| Robin | autonomous | `9.02135e-6` | `2.80305e-7` | `1.09157e-7` |
| Robin | query-time | `8.94209e-3` | `8.94085e-3` | `8.94087e-3` |

The autonomous defect decreases toward the numerical integration floor under
refinement. The query-time defect plateaus near `0.8--0.9%`. This supports the
mechanism interpretation that the query-time maps do not arise from one
shared generator; the observed gap is not explained by giving one path more
RHS evaluations.

## Execution and recovery evidence

The SCNet execution used one RTX 3080 and one process per full cell:

- smoke job `23583314`: `COMPLETED`, exit `0:0`, `16 passed`, 18 smoke cells;
- cache job `23583439`: `COMPLETED`, exit `0:0`, three immutable caches;
- Phase A array `23583444`: 18 cells for seed `31415`, all completed;
- Phase B array `23583659`: 36 cells for seeds `271828` and `161803`, all
  completed;
- aggregate job `23583726`: `COMPLETED`, exit `0:0`, 54 cells accepted.

The aggregate receipt records normal exit, `cell_count=54`,
`boundary_implementation_pass=true`, and `status=passed`. Aggregate artifact
hashes are:

- `aggregate.json`:
  `70804f39499a07dfb59f91c9c8bf0f1c31758e521fb5d1a6ac0b65a8ceb58af4`;
- `cell_summary.csv`:
  `75ef2b92200f382e7519b9fb40c452e7887e66210085e1a27eff54ba2c6dd845`;
- `family_seed_summary.csv`:
  `3f80146ff473335ffd8e4cb094e4a27e0c10a2c7dd1ed9c3608e73a8ab1c263f`;
- aggregate manifest:
  `fc50203ddd5c6ca7e06ceebd602fe29d7a2f9e1cd30331f033b317ad0eeb6250`.

The complete remote canonical root was archived, downloaded in seven chunks,
reassembled, and independently re-hashed. Its archive SHA-256 is
`12c9f38b0b4c4140567b5f6f6612d4dc722b62c65f7d87efb84d1102b95553fe`.
Local recovery rechecked all declared artifact byte counts and SHA-256 values,
all 54 aggregate input receipt hashes, all cache before/after hashes, all
scheduler records, parseability, finite outputs, and empty stderr files.

The recovered evidence is published under
`results/20260830-boundary-family-semigroup-wave2-148f494-r1/`.

## Supported claim

A defensible claim is:

> For the tested one-dimensional autonomous reaction--diffusion problem, a
> parameter-free boundary parameterization defines discrete Dirichlet,
> Neumann, and Robin admissible state spaces and keeps every tested hard-model
> stage and output in the selected space. Defining one state-only autonomous
> generator on each space produces substantially smaller equal-work
> nonuniform composition defect than a parameter- and initialization-matched
> query-time-conditioned vector field. The two structures address distinct
> failure modes and can be composed.

Wave 2 does not establish that:

- the abstract semigroup axiom determines a spatial boundary condition;
- a low composition defect proves agreement with the exact PDE semigroup;
- the result extends to multidimensional or irregular geometries;
- the result preserves mass, energy, a maximum principle, or any unmeasured
  invariant;
- autonomous models universally improve prediction;
- the fixed penalty is the best possible soft-boundary method.

## Main line after Wave 2

The main line is now **boundary-admissible neural semigroups**, not generic
seed accumulation and not the claim that semigroups magically create
boundaries:

\[
\text{spatial domain }X_B
\quad+\quad
\text{tangent autonomous generator}
\quad+\quad
\text{independent boundary/composition/accuracy endpoints}.
\]

The highest-value next test is a conservative no-flux system, such as
Cahn--Hilliard, with a generator of the form

\[
\dot u=-D^\top M_\theta(u)D\mu_\theta(u).
\]

That construction can make no-flux boundary handling and mass conservation
structural, while autonomy supplies temporal composition. A frozen paired
matrix should report boundary-flux residual, mass drift, composition defect,
and prediction error separately. This would test whether the framework
extends from affine endpoint constraints to a physically meaningful invariant
subspace; the current diagonal-mobility model should not be reused unchanged.
