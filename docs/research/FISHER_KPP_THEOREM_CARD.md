# Fisher--KPP theorem card

Date: 2026-08-31

This document instantiates `PDE_THEOREM_CARD.md` for the frozen periodic
Fisher--KPP experiments.  It separates exact PDE facts, fixed-grid neural-flow
facts, conditional transfer statements, and finite-sample diagnostics.  The
card does not reopen or change the locked Section 7.2 Fisher decision.

## A. Continuous initial-boundary-value problem

### A1. Equation and parameters

- **PROVED-IN-REPO:** the target equation is
  \[
  \partial_t u=\nu\partial_{xx}u+r u(1-u),\qquad
  x\in\mathbb T_L,\quad \nu>0,\quad r>0.
  \]
  The frozen experiment uses \(L=10\), \(\nu=0.1\), and \(r=1\).
- **PROVED-IN-REPO:** the equation is autonomous in the physical state \(u\).
- **EXTERNAL-THEOREM:** for periodic initial data of sufficient regularity,
  standard semilinear parabolic theory gives a unique global forward solution
  and continuous dependence.  A publication claim must cite a precise theorem
  and match its state space and regularity hypotheses.

### A2. Correct temporal object

- **PROVED-IN-REPO:** uniqueness and autonomy imply a forward semiflow
  \(S_t\) with \(S_0=I\) and \(S_{t+s}=S_tS_s\) for \(s,t\geq0\).
- **NOT-APPLICABLE:** no backward group is claimed.  Diffusive Fisher--KPP is
  treated only as a forward problem.

### A3. State space and generator domain

- **ASSUMED-FOR-TRANSFER:** one admissible Hilbert setting is
  \(X=L^2_{\mathrm{per}}(0,L)\), with the differential generator on
  \(D(A)=H^2_{\mathrm{per}}(0,L)\).
- **EXTERNAL-THEOREM:** the continuous invariant set is
  \[
  \mathcal K=\{u\in X:0\leq u(x)\leq1\ \text{a.e.}\}.
  \]
  The parabolic comparison principle, using the equilibria 0 and 1, gives
  \(S_t\mathcal K\subseteq\mathcal K\).

### A4. Stability and physical energy

- **PROVED-IN-REPO:** on \(0\leq u,v\leq1\), the reaction secant obeys
  \[
  \bigl(r u(1-u)-r v(1-v)\bigr)(u-v)
  =r\bigl(1-u-v\bigr)(u-v)^2\leq r(u-v)^2.
  \]
  Periodic integration by parts therefore gives the one-sided estimate
  \[
  \langle A(u)-A(v),u-v\rangle_{L^2}
  \leq r\lVert u-v\rVert_{L^2}^2.
  \]
  This is an upper stability bound, not a contraction claim.
- **PROVED-IN-REPO:** the Fisher physical energy
  \[
  \mathcal E(u)=\int_0^L\left(\frac{\nu}{2}|u_x|^2-
  \frac r2u^2+\frac r3u^3\right)\,dx
  \]
  has variational derivative
  \(\delta\mathcal E/\delta u=-\nu u_{xx}-ru+ru^2\), so sufficiently regular
  solutions satisfy
  \[
  \frac{d}{dt}\mathcal E(u(t))=-\lVert u_t(t)\rVert_{L^2}^2\leq0.
  \]
  This physical energy is distinct from the checkpoint's learned latent
  energy and from a plain \(L^2\) norm.

## B. Periodic boundary and invariant-region structure

- **PROVED-IN-REPO:** the spatial boundary law is periodic.  The grid stores
  \(N\) distinct samples on \([0,L)\); it does not duplicate both endpoints.
  Consequently, a Dirichlet-style endpoint residual is not a meaningful
  Fisher metric.
- **VERIFIED-ALGEBRAICALLY:** the reference spectral generator and the learned
  interaction masks use periodic Fourier/circular operations.
- **OPEN:** the Fourier collocation method-of-lines system and the finite-step
  ETD--RK4 reference integrator are not proved to preserve the nodal cube
  \([0,1]^N\).  Continuous maximum-principle invariance must not be silently
  transferred to this spectral discretization.
- **PROVED-IN-REPO:** Model A's exact learned latent ODE trajectory, after
  sigmoid decoding, lies in \((0,1)^N\) because \(u=\sigma(z)\).  This is
  decoder-image invariance, not
  a discrete maximum principle for the Fisher generator.
- **OPEN:** neither A nor B is proved to dissipate the Fisher physical energy.
  Both dissipate their own learned latent energy for a fixed vector field.

## C. Reference method of lines

### C1. Grid and norm

- **PROVED-IN-REPO:** \(x_j=jL/N\), \(j=0,\ldots,N-1\), with
  \(X_h=\mathbb R^N\), \(\Delta x=L/N\), and
  \[
  \lVert v\rVert_h^2=\Delta x\sum_{j=0}^{N-1}v_j^2.
  \]
- **ASSUMED-FOR-TRANSFER:** trigonometric interpolation is the reconstruction
  \(I_h\); a mesh-dependent spatial convergence estimate is not established
  by the current experiment.

### C2. Semidiscrete generator

- **PROVED-IN-REPO:** the diagnostic reference generator is the Fourier
  collocation vector field
  \[
  A_h(v)=\nu D_{xx,h}^{\mathrm{spec}}v+r v\odot(1-v).
  \]
- **VERIFIED-ALGEBRAICALLY:** the spectral Laplacian is symmetric negative
  semidefinite in the uniform-grid inner product.  For \(v,w\in[0,1]^N\),
  \[
  \langle A_h(v)-A_h(w),v-w\rangle_h
  \leq r\lVert v-w\rVert_h^2.
  \]
- **OPEN:** a mesh-uniform spatial error
  \(\varepsilon_{\mathrm{space}}(h)\) has not yet been proved or measured.

## D. Frozen learned generators

### D1. Physical-coordinate vector fields

Let \(E(u)=\operatorname{logit}(\operatorname{clamp}_\epsilon u)\),
\(D(z)=\sigma(z)\), let \(f_{\theta}(z)\) be the latent vector field, and let
\(J_D(z)=\operatorname{diag}(D(z)\odot(1-D(z)))\).  For a raw sampled state
\(u\), define its represented state \(v=D(E(u))\).  The sample-indexed
physical-coordinate vector measured by the new evaluator is
\[
G_{\theta,h}[u]=J_D(E(u))f_\theta(E(u)).
\]
On a clamp-inactive represented set, \(E(D(z))=z\), so this agrees with the
single-valued physical vector field
\(F_{\theta,h}^{u}(v)=J_D(E(v))f_\theta(E(v))\).  Outside that set the
finite-sample diagnostic remains well defined as \(G_{\theta,h}[u]\), but it
must not be silently promoted to a vector field depending only on \(v\).

- **PROVED-IN-REPO:** for Model A, \(f_\theta\) is state-only and autonomous.
  Its exact latent ODE defines one forward semiflow \(\phi_t\).
- **CONDITIONAL:** the physical map \(\Phi_t=D\phi_tE\) inherits
  \(\Phi_t\Phi_s=\Phi_{t+s}\) only on trajectories for which
  \(E(D(z))=z\).  With the implemented clamped logit encoder, this requires
  every decoded latent state to remain in
  \([\epsilon,1-\epsilon]^N\).  Outside that region the decoded/re-encoded
  map has only an empirically testable composition property.
- **PROVED-IN-REPO:** for Model B, the requested query duration \(\tau\) enters
  the mobility, producing \(F_{\theta,h}^{u,[\tau]}\).  Each frozen value of
  \(\tau\) defines a vector field, but cross-query maps need not share one
  generator.
- **OPEN:** no uniform generator-matching bound has been proved for either
  model.
- **OPEN:** no uniform one-sided Lipschitz constant has been proved for either
  model on the common trajectory set.

### D2. Quantities that the new experiment may estimate

- **VERIFIED-NUMERICALLY:** on a frozen finite raw-state set
  \(U_h^{\mathrm{eval}}\), the evaluator forms
  \(V_h^{\mathrm{eval}}=D(E(U_h^{\mathrm{eval}}))\), reports the
  representation error \(\lVert D(E(u))-u\rVert_h\), and may report
  \[
  \widehat\varepsilon_{\mathrm{gen}}(V_h^{\mathrm{eval}})
  =\max_{v\in V_h^{\mathrm{eval}}}
  \lVert G_{\theta,h}[u]-A_h(v)\rVert_h.
  \]
- **OPEN:** this is a represented-state diagnostic.  It is not a residual at
  the unrepresented raw state when \(D(E(u))\ne u\), and it does not by itself
  define the common invariant set required by the transfer theorem.
- **VERIFIED-NUMERICALLY:** on frozen pairs \(\mathcal P_h\), it may report
  \[
  \widehat\omega_h=
  \max_{(v,w)\in\mathcal P_h}
  \frac{\langle F(v)-F(w),v-w\rangle_h}
  {\lVert v-w\rVert_h^2}.
  \]
- **OPEN:** neither finite maximum is a supremum over a continuum.  These
  estimates cannot be substituted into a theorem as certified uniform
  constants without an additional covering, interval-bound, or analytic
  argument.

## E. Time integration

- **PROVED-IN-REPO:** the production learned-flow integrator is classical RK4
  with 30 substeps per requested map.
- **CONDITIONAL:** while neither the latent clamp nor the encoder clamp is
  active and the trajectory stays in a smooth bounded set, standard RK4
  theory gives fourth-order global time error for each fixed learned vector
  field.  If the projection is active, the implementation is projected RK4
  and the standard order statement does not apply without a separate proof.
- **VERIFIED-NUMERICALLY:** the new evaluator compares production and refined
  repeated-lag rollouts against a 120-substep-per-call surrogate.  This
  estimates learned-flow time-discretization error; it is not an error against
  the exact PDE.
- **OPEN:** if latent values hit the hard clamp at \(\pm20\), or if decoding
  followed by encoding activates the encoder clamp, the smooth-vector-field
  RK4 and decoded-semigroup arguments do not directly describe the implemented
  map.  The evaluator must report both events and the re-encoding discrepancy.

## F. Conditional transfer theorem instance

On a clamp-inactive common set \(V_h\), define the single-valued physical field
\(F_{\theta,h}^{u}(v)=J_D(E(v))f_\theta(E(v))\).  Let \(x'=A_h(x)\) and
\(y'=F_{\theta,h}^{u}(y)\) remain in \(V_h\).  If
\[
\sup_{v\in V_h}\lVert F_{\theta,h}^{u}(v)-A_h(v)\rVert_h
\leq\varepsilon_{\mathrm{gen},h}
\]
and
\[
\langle F_{\theta,h}^{u}(v)-F_{\theta,h}^{u}(w),v-w\rangle_h
\leq\omega_h\lVert v-w\rVert_h^2,
\]
then the theorem in `boundary_admissible_semigroup_theory.tex` yields
\[
\lVert y(t)-x(t)\rVert_h
\leq e^{\omega_ht}\lVert y(0)-x(0)\rVert_h
+\varepsilon_{\mathrm{gen},h}\chi_{\omega_h}(t).
\]
After adding spatial reconstruction and numerical integration errors, the full
layered bound is
\[
\lVert I_h\widehat y(T)-S_Tu_0\rVert_X
\leq\varepsilon_{\mathrm{space}}(h)+C_I\left[
\eta_{\mathrm{time}}(T)+e^{\omega_hT}\delta_{0,h}
+\varepsilon_{\mathrm{gen},h}\chi_{\omega_h}(T)\right].
\]
The present experiment samples represented reference and learned trajectory
states.  The union of those represented samples is its observed state set; it
is not proved invariant or equal to the theorem's common set \(V_h\).  The
experiment estimates finite-sample analogues of \(\eta_{\mathrm{time}}\),
\(\varepsilon_{\mathrm{gen},h}\), and \(\omega_h\); it does not certify the
uniform assumptions.

## G. Issued claim boundary

- **Architecture-level theorem:** Model A's latent ODE is one autonomous
  finite-dimensional flow and has an exact continuous-time composition law.
  Its sigmoid-decoded state remains in \((0,1)^N\).  The decoded/re-encoded
  physical map inherits the law only where \(E\circ D=I\) along the trajectory.
- **Conditional numerical theorem:** classical RK4 converges with order four
  for a fixed smooth learned vector field while relevant trajectories avoid
  the clamp and remain in a common bounded smoothness region.
- **Conditional PDE-transfer theorem:** the layered bound above holds if
  spatial consistency, uniform generator matching, and uniform one-sided
  stability are supplied.
- **Empirical result:** the locked Fisher study established a much smaller
  equal-work composition defect for A but no material cross-seed MSE advantage.
- **Empirical diagnostic:** the frozen checkpoint-only generator study found
  essentially tied generator residuals (geometric-mean A/B `0.9990` at both
  query lags), a two-of-three-seed empirical one-sided-stability ordering for
  A, and non-dominant learned-flow RK4 error.  Thus the observed composition
  advantage did not reduce every term required by the conditional transfer
  bound; see `FISHER_GENERATOR_STABILITY_RESULTS.md`.
- **Not proved:** a mesh-uniform Fisher transfer theorem, a discrete spectral
  maximum principle, Fisher-energy decay for the learned models, or a uniform
  generator/stability bound.
