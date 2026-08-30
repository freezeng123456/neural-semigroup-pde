# PDE theorem card

This card is the analysis gate for extending the boundary-admissible neural-semigroup programme to a new PDE. It prevents three distinct statements from being conflated:

1. an autonomous finite-dimensional neural ODE has an exact forward-flow composition law;
2. a numerical integrator approximately composes that flow;
3. the learned flow approximates the solution operator of a specified PDE.

Only the first statement follows from autonomy and ODE well-posedness. The third requires a PDE-specific state space, boundary domain, semidiscrete consistency, generator matching, and stability estimate. A new PDE should not enter a transfer experiment until every required item below is either proved, supported by a precise reference, or explicitly marked as an unresolved obligation.

## Status labels

Every line must use one of these labels:

- PROVED-IN-REPO: a complete proof is present in a named repository file;
- EXTERNAL-THEOREM: a precise theorem and all hypothesis matches are cited;
- VERIFIED-ALGEBRAICALLY: an exact finite-dimensional identity is checked from the implemented operators;
- VERIFIED-NUMERICALLY: an empirical diagnostic only, with protocol and tolerance;
- ASSUMED-FOR-TRANSFER: an explicit hypothesis of a conditional theorem;
- OPEN: not established and therefore unavailable for a transfer claim;
- NOT-APPLICABLE: accompanied by a reason.

An experimental pass cannot upgrade ASSUMED-FOR-TRANSFER or OPEN to a theorem.

## A. Continuous initial-boundary-value problem

### A1. Equation and parameters

- PDE:
- Spatial domain:
- Parameter set:
- Initial-data class:
- Forcing and boundary data:
- Autonomous in the stated state variables? yes / no

### A2. Correct temporal object

- If autonomous: define the proposed forward semiflow \(S_t:X\to X\).
- If explicitly time dependent: define the evolution family \(U(t,s)\) and its cocycle law \(U(t,r)U(r,s)=U(t,s)\).
- If time is augmented into the state: define the augmented state, its norm, and the projection back to the physical state.
- State whether backward solutions exist. Do not call a forward dissipative semiflow a group without invertibility.

### A3. State space and generator domain

- Banach or Hilbert state space \(X\):
- Norm or inner product:
- Continuous boundary operator \(B\):
- Generator domain \(D(A)\), including the boundary condition:
- Admissible or invariant set \(\mathcal K\):

### A4. Well-posedness

- Existence:
- Uniqueness:
- Global forward existence:
- Continuous dependence:
- Lipschitz, one-sided Lipschitz, contractive, or monotone estimate:
- Proof or exact external theorem:
- Verification that all theorem hypotheses match this PDE, data class, and boundary condition:

## B. PDE-specific structure

### B1. Spatial boundary law

- Continuous boundary condition:
- Is it Dirichlet, Neumann/no-flux, Robin, periodic, dynamic, or mixed?
- Is the boundary data time independent?
- Compatibility condition between initial and boundary data:
- Boundary contribution in the energy, mass, or integration-by-parts identity:

### B2. Invariant region or maximum principle

- Proposed invariant set:
- Comparison or maximum-principle argument:
- Restrictions on reaction, forcing, boundary data, and time interval:
- Does the property apply to the exact PDE, the semidiscretization, the neural generator, or only observed trajectories?

### B3. Conservation or Lyapunov law

- Physical conserved quantity:
- Physical energy or entropy:
- Exact identity or inequality:
- Required generator geometry:
  - \(L^2\) gradient flow;
  - conservative \(H^{-1}\)-type flow;
  - conservative flux form;
  - skew/transport plus dissipation;
  - Hamiltonian, symplectic, or other.
- Distinguish the physical functional from any learned latent energy.

## C. Discrete boundary and method of lines

### C1. Grid and discrete state space

- Grid and geometry:
- Discrete space \(X_h\):
- Mesh-dependent norm \(\|\cdot\|_h\):
- Sampling/projection \(P_h:X\to X_h\):
- Reconstruction \(I_h:X_h\to X\):
- Reconstruction stability constant \(C_I\):

### C2. Discrete boundary operator

- Discrete affine relation \(C_hu_h=b_h\):
- Boundary-admissible set \(M_{B,h}\):
- State reconstruction \(\Pi_{B,h}\):
- Tangent reconstruction \(Q_{B,h}\):
- Exact identities to verify:

  \[
  C_h\Pi_{B,h}u=b_h,\qquad C_hQ_{B,h}v=0,\qquad
  \Pi_{B,h}^2=\Pi_{B,h},\qquad Q_{B,h}^2=Q_{B,h}.
  \]

- State clearly whether these maps are merely retractions or are orthogonal in a specified inner product.

### C3. Continuous-boundary consistency

- Grid convention and one-sided, ghost-point, finite-volume, or trace formula:
- Consistency statement \(C_hP_hu\to Bu\):
- Rate and required regularity:
- Stability/nondegeneracy condition:
- For Robin data, distinguish algebraic endpoint invertibility from well-posedness of the continuous boundary problem.

### C4. Method-of-lines generator

- Reference semidiscrete generator \(A_h:X_h\to X_h\):
- Local/global well-posedness:
- Common invariant set \(V_h\):
- Discrete boundary tangency \(C_hA_h(v)=0\):
- Discrete invariant-region, mass, and energy identities:
- Spatial convergence estimate:

  \[
  \sup_{0\le t\le T}\|I_hS_t^h(P_hu_0)-S_tu_0\|_X
  \le \varepsilon_{\mathrm{space}}(h).
  \]

## D. Learned generator

### D1. Architecture

- Learned generator \(F_{\theta,h}\):
- Does requested duration enter the vector field?
- If autonomous, prove global or trajectory-local well-posedness.
- If query-time conditioned, state that different durations need not share a generator.
- Exact architectural properties:
- Properties imposed only by loss terms:
- Properties measured only after training:

### D2. Boundary invariance

- Prove \(C_hF_{\theta,h}(u)=0\) on the relevant state set.
- Prove exact-flow invariance of \(M_{B,h}\).
- Prove stage preservation for the selected integrator, or label it numerical only.
- Record floating-point tolerances separately from exact algebra.

### D3. Generator matching

- Common trajectory set \(V_h\) containing both learned and reference trajectories:
- Uniform generator error:

  \[
  \sup_{v\in V_h}\|F_{\theta,h}(v)-A_h(v)\|_h
  \le \varepsilon_{\mathrm{gen},h}.
  \]

- How the error is bounded: theorem, approximation result, validation estimate, or assumption.
- Confirm that no test cache or selected horizon is used to choose the checkpoint or this bound.

### D4. Stability

- One-sided Lipschitz or monotonicity estimate:

  \[
  \langle F_{\theta,h}(v)-F_{\theta,h}(w),v-w\rangle_h
  \le \omega_h\|v-w\|_h^2.
  \]

- Is \(\omega_h\) positive, zero, or negative?
- Is the bound uniform in \(h\), parameters, and the relevant state set?
- If unavailable, long-time transfer remains OPEN even if composition defect is small.

## E. Time integration and composition

### E1. Integrator

- Method and formal order:
- Step-size or function-evaluation budget:
- Smoothness and stability hypotheses:
- Effect of state reconstruction on the method:
- Global time-discretization bound \(\eta_{\mathrm{time}}(T)\):

### E2. Composition metric

- Direct partition:
- Composed partition:
- Equal-work rule:
- Absolute defect:
- Relative denominator and floor:
- Temporal refinement grid:
- Expected order and acceptance interval fixed before reading test results:

For autonomous RK4, a valid conditional target is

  \[
  \|\Psi_\pi(T,u)-\Psi_\rho(T,u)\|_h
  \le C_TT\bigl(|\pi|^4+|\rho|^4\bigr).
  \]

This is a numerical-flow statement. It is not generator matching and it is not a PDE error bound.

## F. Full transfer statement

Write the exact theorem instance before launching the experiment. A valid layered target has the form

  \[
  \|I_h\widehat y(T)-S_Tu_0\|_X
  \le \varepsilon_{\mathrm{space}}(h)
  +C_I\left[\eta_{\mathrm{time}}(T)
  +e^{\omega_hT}\delta_{0,h}
  +\varepsilon_{\mathrm{gen},h}\chi_{\omega_h}(T)\right],
  \]

where

  \[
  \chi_\omega(T)=
  \begin{cases}
  (e^{\omega T}-1)/\omega,&\omega\ne0,\\
  T,&\omega=0.
  \end{cases}
  \]

For \(\omega_h=-\lambda_h<0\), the generator-mismatch contribution is bounded by \(\varepsilon_{\mathrm{gen},h}/\lambda_h\). If any term is only assumed or empirically estimated, label the final result conditional.

## G. Falsification and experiment gate

- Primary theoretical prediction:
- Negative control:
- Metrics kept logically separate:
  - boundary or flux residual;
  - invariant, mass, or energy violation;
  - absolute and relative numerical composition defect;
  - generator residual, when measurable;
  - one-step and long-horizon PDE error.
- Predeclared pass/fail or advance rule:
- Result that would falsify the proposed mechanism:
- Result that would show the architecture is out of class:
- Formal versus exploratory evidence label:
- Frozen source, cache, seed, checkpoint-selection, and result-root rules:

## H. PDE-family routing guide

| PDE class | Correct temporal object | Essential PDE structure | Minimum architecture before transfer |
|---|---|---|---|
| Fisher--KPP / autonomous reaction--diffusion | Forward semiflow | Compatible boundary domain; positivity or invariant interval under stated assumptions | Boundary-tangent diffusion plus a proved invariant-region or comparison mechanism |
| Allen--Cahn | Forward semiflow | Physical free-energy decay | Discrete \(L^2\) gradient structure matched to the physical energy |
| Cahn--Hilliard | Forward semiflow | Mass conservation and free-energy decay | Conservative \(-D_h^\top M_hD_h\) mobility and compatible periodic/no-flux operators |
| Viscous Burgers | Forward semiflow | Conservative transport and viscous dissipation | Conservative/skew transport component plus dissipative diffusion |
| Explicitly nonautonomous PDE | Evolution family \(U(t,s)\) | Absolute clock and start-time dependence | Two-time evolution model or a correctly augmented autonomous state |

## I. Claim issued after completing the card

Fill in exactly one statement at each level:

- **Architecture-level theorem:**
- **Conditional numerical theorem:**
- **Conditional PDE-transfer theorem:**
- **Empirical result:**
- **Not proved / excluded from the claim:**

The default claim boundary is:

> A hard affine boundary reconstruction and tangent autonomous generator define a boundary-preserving finite-dimensional forward semiflow. Agreement with a target PDE solution operator additionally requires PDE-specific spatial consistency, generator matching, and stability; numerical composition consistency alone is insufficient.
