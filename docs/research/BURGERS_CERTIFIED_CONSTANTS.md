# Certified constants for the minimal Burgers generator

Date: 2026-09-05

`FISHER_KPP_THEOREM_CARD.md` marks the generator-matching bound and the
one-sided Lipschitz constant `OPEN`, and records why the measured versions
cannot be promoted:

> neither finite maximum is a supremum over a continuum.  These estimates
> cannot be substituted into a theorem as certified uniform constants without
> an additional covering, interval-bound, or analytic argument.

The generator adopted in `BURGERS_MINIMAL_GENERATOR_AB_RESULTS.md` is small
enough to supply those arguments.  Its flux is a pointwise one-hidden-layer
tanh map with 33 parameters, so its derivative bounds are analytic, and a
covering argument converts a grid extremum into a genuine supremum.  This note
derives the constants, evaluates them from the committed weights, instantiates
the layered rollout bound, and reports how loose it is and why.

All values are `exploratory=true` and `do_not_use_for_formal=true`.  Nothing
here changes any experimental result.

## Setting

Model A's generator on the periodic grid is
$F_\theta(u)=-D_0f_\theta(u)+\nu D_2u$, where $D_0$ is the centered periodic
divergence, $D_2$ the three-point periodic Laplacian, and $f_\theta$ acts
componentwise.  The finite-difference reference generator on the same grid and
with the same operators is $A_h(u)=-D_0\bigl(u^2/2\bigr)+\nu D_2u$.  The mesh
norm is $\lVert v\rVert_h^2=\Delta x\sum_jv_j^2$.

## Propositions

**P1, exact scalar reduction.**  With patch radius `0` the flux network reads
the pair $(u_j,c)$, and model A sets $c\equiv0$.  Therefore
$f_\theta(s)=W_2\tanh(ws+b_1)+b_2$ with $w=W_1e_1$.  This is an identity, not
an approximation; the discarded control column is recorded in the artifact so
the reduction is auditable.

**P2, certified derivative bounds.**  Differentiating P1 gives
$f_\theta'(s)=\sum_jW_{2,j}w_j\operatorname{sech}^2(w_js+b_{1,j})$.  Since
$0<\operatorname{sech}^2\le1$,

$\sup_s|f_\theta'(s)|\le\sum_j|W_{2,j}w_j|=:L^{\mathrm{prod}}$.

Writing $t=\tanh x$ gives
$\frac{d}{dx}\operatorname{sech}^2x=-2t(1-t^2)$, whose modulus is maximal at
$t=1/\sqrt3$ with value $\kappa=4/(3\sqrt3)$.  Hence
$\sup_s|f_\theta''(s)|\le\kappa\sum_j|W_{2,j}|w_j^2=:L_2$, and on a uniform
grid of spacing $h$ covering $[-M,M]$,

$\sup_{|s|\le M}|f_\theta'(s)|\le\max_{\mathrm{grid}}|f_\theta'|+L_2h/2=:L^{\mathrm{cov}}$.

Both bounds are certified; the run uses $L_f=\min(L^{\mathrm{prod}},L^{\mathrm{cov}})$.

**P3, exact operator norms and symmetry.**  Both operators are circulant with
eigenvalues $\mathrm i\sin(2\pi k/N)/\Delta x$ and
$(2\cos(2\pi k/N)-2)/\Delta x^2$, so
$\lVert D_0\rVert_2=\max_k|\sin(2\pi k/N)|/\Delta x$ and
$\lVert D_2\rVert_2=\max_k(2-2\cos(2\pi k/N))/\Delta x^2$.  Moreover
$D_0^\top=-D_0$ and $D_2=D_2^\top\preceq0$.  The artifact verifies all of this
against explicitly assembled matrices: the skew-symmetry residual and the
symmetry residual are exactly `0.0`, the largest Laplacian eigenvalue is
`-6.26e-14`, and both norms agree with `torch.linalg.matrix_norm` to `1e-10`.

**P4, certified Lipschitz constant.**  Componentwise application of an
$L_f$-Lipschitz scalar map is $L_f$-Lipschitz in the mesh norm, so
$\operatorname{Lip}(F_\theta)\le\lVert D_0\rVert_2L_f+\nu\lVert D_2\rVert_2$.

**P5, certified one-sided constants.**  For $e=u-v$, skewness of $D_0$ gives
$\langle-D_0g,e\rangle_h=\langle g,D_0e\rangle_h$, and $D_2\preceq0$ gives
$\nu\langle D_2e,e\rangle_h\le0$.  Hence

$\langle F_\theta(u)-F_\theta(v),e\rangle_h\le L_f\lVert D_0\rVert_2\lVert e\rVert_h^2$,

so $\omega_\theta\le\lVert D_0\rVert_2L_f$.  The same argument applied to
$A_h$, whose flux $s^2/2$ has derivative bounded by $M$ on $[-M,M]$, gives
$\omega_A\le\lVert D_0\rVert_2M$.

**P6, gauge-invariant generator matching.**  $D_0\mathbf 1=0$, so for every
constant $c$,
$F_\theta(u)-A_h(u)=-D_0\bigl(f_\theta(u)-u^2/2-c\mathbf 1\bigr)$.  The flux is
therefore only determined up to an additive constant and the correct certified
quantity is the variation, not the raw supremum:

$\varepsilon_f:=\min_c\sup_{|s|\le M}\bigl|f_\theta(s)-s^2/2-c\bigr|$,

which equals half the range of $f_\theta(s)-s^2/2$ and is certified by the
same covering with derivative bound $L_f+M$.  Then, since a constant vector
has mesh norm $c\sqrt L$,

$\sup_{\lVert u\rVert_\infty\le M}\lVert F_\theta(u)-A_h(u)\rVert_h\le\lVert D_0\rVert_2\sqrt L\,\varepsilon_f=:\varepsilon_{\mathrm{gen}}$.

Getting this wrong is not academic.  The first run of the artifact omitted the
gauge and reported $\varepsilon_f=2.3196$; the optimal constant turns out to be
$-2.2996$, so almost the whole apparent error was an unobservable offset and
the certified bound was inflated by a factor of about `116`.

**P7, layered bound.**  With both exact flows started from the same state, the
trajectory-localized Grönwall argument of the theorem card's Section F2 gives

$\lVert y(T)-x(T)\rVert_h\le e^{\omega_AT}\lVert y(0)-x(0)\rVert_h+\varepsilon_{\mathrm{gen}}\chi_{\omega_A}(T)$,
$\qquad\chi_\omega(T)=(e^{\omega T}-1)/\omega$.

## Certified values

Evaluated from the three committed model-A checkpoints, on the interval
$|u|\le M$ with $M$ the observed trajectory maximum plus a `0.05` margin.  The
covering uses `2,000,001` points.

| Quantity | `31415` | `271828` | `161803` |
|---|---:|---:|---:|
| State interval $M$ | `1.1001` | `1.0984` | `1.1004` |
| $L^{\mathrm{prod}}$, product bound | `2.4125` | `2.3818` | `2.4777` |
| $L^{\mathrm{cov}}$, covering bound | `0.8934` | `0.9243` | `0.8965` |
| $\varepsilon_f$, gauge-invariant | `0.019998` | `0.021425` | `0.019245` |
| $\operatorname{Lip}(F_\theta)$ | `13.250` | `13.564` | `13.282` |
| $\omega_\theta$, learned | `9.100` | `9.414` | `9.132` |
| $\omega_A$, reference | `11.206` | `11.188` | `11.209` |
| $\varepsilon_{\mathrm{gen}}$ | `0.5106` | `0.5470` | `0.4914` |

The covering bound beats the product bound by a factor of `2.6` to `2.7`, so
the covering argument is worth making rather than settling for the analytic
product.

## The bound against the observed error

The observed error is the mesh norm between the learned exact flow and the
finite-difference reference exact flow, both integrated from the same 50
validation initial conditions with RK4 at `10,000` substeps per unit time.

| Seed | Horizon | Certified bound | Observed max | Looseness |
|---:|---:|---:|---:|---:|
| `31415` | `0.4` | `3.984` | `0.029945` | `133` |
| `31415` | `0.8` | `356.3` | `0.045820` | `7776` |
| `271828` | `0.4` | `4.245` | `0.029050` | `146` |
| `271828` | `0.8` | `377.0` | `0.044620` | `8449` |
| `161803` | `0.4` | `3.838` | `0.029860` | `129` |
| `161803` | `0.8` | `343.6` | `0.045730` | `7515` |

## Where the looseness is, which is the useful part

Decomposing the bound at seed `31415` — the other two seeds agree to within a
few percent:

| Variant | `T=0.4` | `T=0.8` |
|---|---:|---:|
| Certified bound | `3.984` | `356.3` |
| Same bound with $\omega=0$ | `0.2042` | `0.4085` |
| Same bound with the sampled rate `0.6343`, **not certified** | `0.2325` | `0.5321` |
| Observed | `0.029945` | `0.045820` |

The exponential amplification costs a factor of about `20` at `T=0.4` and
about `870` at `T=0.8`; the generator-matching term costs only about `7` and
`9`.  The certified $\omega_A=11.206$ is `17.7` times larger than the largest
one-sided quotient actually attained on sampled pairs of reference snapshots,
`0.6343`, and it enters the bound exponentially.

**So essentially all of the looseness is in the one-sided Lipschitz constant,
not in generator matching.**  With a sharp rate the certified bound would be
about `8` times the observed error at `T=0.4` and about `12` times at `T=0.8`,
which would be a usable a priori estimate.  With the crude rate it is `133`
and `7776` times.

The crude rate is crude for a clear reason: $\omega_A\le\lVert D_0\rVert_2M$
discards the conservative structure of the centered flux divergence entirely.
In the continuous setting the corresponding estimate integrates by parts twice
and yields a constant proportional to $\lVert\partial_x(u+v)\rVert_\infty$
rather than to $\lVert\partial_x\rVert\cdot\lVert u\rVert_\infty$.  A discrete
analogue in split or skew form is the concrete way to close the gap; it is not
attempted here because writing down a discrete summation-by-parts identity for
the plain centered divergence form requires care and an unverified identity
would be worse than a crude bound.

## What is not certified

- **The state interval is observed, not proved invariant.**  This generator
  has no bounded decoder, so nothing structurally confines $u$ to $[-M,M]$.
  Every constant above is certified *on* that interval; the bound is
  conditional on both trajectories remaining in it, which they did.
- **Time-integration error is excluded.**  The bound compares exact flows.
  The production map is RK4 with 12 substeps per call, and its error is a
  separate measured term.
- **Finite-difference versus spectral spatial consistency is excluded.**  The
  reference here is the finite-difference semidiscrete generator, chosen so
  that the comparison isolates the learned flux.  The gap to the spectral
  reference solver used to generate the data is a standard discretization
  term and is not estimated.

## Status and next steps

For this model, on this interval, the two constants that
`FISHER_KPP_THEOREM_CARD.md` lists as `OPEN` with only finite-sample estimates
are now certified, and the layered bound is a real number rather than a
schema.  That is the first time in this project that a rollout error bound has
been instantiated without a sampled constant standing in for a supremum.

Three next steps, in order of value:

1. **A sharp discrete one-sided estimate.**  Everything else is already tight
   enough; this single constant is worth about three orders of magnitude at
   `T=0.8`.
2. **A structural state bound.**  Without one, the theorem stays conditional
   on an observed interval.  The bounded-decoder construction used on Fisher
   supplies exactly this, at the cost of the divergence form; whether the two
   can be combined is open.
3. **Promotion into the LaTeX manuscript.**  P1--P7 are written to be
   transcribed into a fragment alongside `boundary_admissible_semigroup_theory.tex`,
   which the main note already `\input`s.  This was not done here because the
   execution host has no TeX engine, and the project rule is not to claim a
   compile that was not run.

Machine-readable constants, self-checks, sampled diagnostics, and the bound
decomposition for all three seeds are stored under
`results/20260905-burgers-certified-constants-r1/`.
