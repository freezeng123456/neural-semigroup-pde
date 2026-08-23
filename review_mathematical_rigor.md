# Mathematical Rigor Review: Deep Operator Learning of Maximum-Principle-Preserving Dissipative Semigroups

**Reviewer assessment**: The paper presents a well-structured framework with a clear logical arc from PDE-level specification to architectural verification to transfer theorem. The core ideas are sound and the main architectural guarantees (Proposition 5.4) are correctly established. However, there are several mathematical issues ranging from proof gaps to notation inconsistencies that need to be addressed before publication.

---

## CRITICAL Issues

### [CRITICAL] [Section 8, Lines 957-975] Transfer Theorem proof drops semigroup defect term and conflates continuous/discrete flows

The proof of Theorem 8.1, Part (iii), bound on (B) (line 960) states:

> "By Proposition 8.4, $\|S_{n\Delta t}^N u_0^N - \cS_{n\Delta t}u_0\| \le C(N)$."

However, Proposition 8.4 (line 926) actually gives $\|S_t^N u_0^N - \cS_tu_0\| \le C(N) + \varepsilon_{\mathrm{sg}}^N(t)$. The term $\varepsilon_{\mathrm{sg}}^N(t)$ is dropped without justification.

**Root cause**: The notation $S_t^N$ is ambiguous throughout Section 8. Assumption 8.3 defines $S_{\Delta t}^N$ as a one-step integrator (line 914) but then uses $S_t^N$ in the convergence clause (line 919) without specifying whether this is:
- (a) the exact continuous-time flow of the semidiscrete ODE $\partial_t u_N = \cF_N(u_N)$, or
- (b) the iterated discrete integrator $(S_{\Delta t}^N)^{\circ \lfloor t/\Delta t\rfloor}$.

If interpretation (a): $\varepsilon_{\mathrm{sg}}^N = 0$ (exact flow has exact semigroup property), and the bound $C(N)$ is correct. But then the "one-step increment" Gronwall argument on lines 962-964 (comparing $\PhiT$ and $S^N$ step by step in increments of $\Delta t$) is conceptually wrong—both flows are continuous and should be compared directly, not step-by-step.

If interpretation (b): $\varepsilon_{\mathrm{sg}}^N > 0$ (discrete integrator introduces semigroup defect), and the dropped term is a genuine error.

**Fix suggestion**: Clearly define $\tilde{S}_t^N$ as the exact continuous-time semidiscrete flow and $S_{\Delta t}^N$ as the discrete integrator. State and prove two separate bounds:
- For the continuous-to-continuous comparison: use a standard ODE comparison (Gronwall on the vector field difference) between $\PhiT$ (continuous latent flow) and $\tilde{S}_t^N$ (continuous semidiscrete flow), with the one-step argument replaced by a continuous-time Gronwall.
- Then add the integrator truncation error $\varepsilon_{\mathrm{sg}}^N$ if comparing to the discrete iterated scheme.

---

## HIGH Issues

### [HIGH] [Section 8, Lines 906-921] Assumption 8.3 conflates semidiscrete ODE and discrete integrator

Assumption 8.3 introduces both the semidiscrete ODE (line 912) and a one-step integrator $S_{\Delta t}^N$ (line 914), but the convergence clause (line 919) uses $S_t^N$ without distinguishing which object it refers to. The structural preservation clauses (a) and (b) are stated for the discrete integrator $S_{\Delta t}^N$, while consistency (clause (i)) and convergence (clause (iii)) naturally refer to the continuous semidiscrete flow.

**Fix suggestion**: Split the assumption into two parts:
- Assumption 8.3a (Semidiscrete ODE): consistency and convergence of the continuous semidiscrete flow $\tilde{S}_t^N$.
- Assumption 8.3b (Discrete integrator): invariance, dissipation, and truncation error order of the one-step map $S_{\Delta t}^N$.

### [HIGH] [Section 9, Lines 990-1015] Proposition 9.1 is a proof sketch, not a proof; the universal approximation claim requires additional justification

Proposition 9.1 claims universal approximation for the generator matching condition (equation 9.1), but:

1. The proof is labeled "Proof sketch" and consists of five informal steps. The key Step 4 (composition approximation, lines 1007-1011) asserts that because $K_\theta$ and $\nabla\Psi_\theta$ are individually universal approximators, their product $K_\theta \cdot \nabla\Psi_\theta$ is dense in the target class. This does not follow from standard universal approximation theorems—the product of two approximable functions need not approximate the product of the targets unless the approximation is simultaneous (same parameter $\theta$ for both factors).

2. The class of realizable generators is restricted: the architecture produces vector fields of the form $\mathrm{D}g \cdot K \cdot \nabla\Psi$ with $K$ PSD. The proposition restricts to generators whose pullback $\mathrm{D}g^{-1} \cdot \cF_N$ is a gradient field, but the proof does not establish that this restriction is necessary. More importantly, even within this class, the simultaneous approximation of $K_{\mathrm{target}}$ and $\nabla\varphi_{\mathrm{target}}$ by a single parameter $\theta$ is not proven.

3. Step 2 (interaction coefficients, line 1003) claims PSD completion for general nearest-neighbor graphs but does not provide a proof or reference.

**Fix suggestion**: Either provide a complete proof or weaken the statement to clearly delineate what is proven (individual component approximation) from what requires further work (compositional expressivity).

### [HIGH] [Section 5, Lines 380-382] All experiments use $\beta_V=0$, violating the global-flow assumption

Remark 5.5 honestly acknowledges that $\beta_V=0$ in the experiments, which means the coercivity condition (equation 5.5) is not satisfied and Proposition 5.2 does not apply. Since Proposition 5.2 (global flow existence) is the foundation for all architectural guarantees (Proposition 5.4, Corollary 5.6), none of the theoretical guarantees formally apply to the experimental configuration.

This is a significant gap between theory and practice. The paper notes this is "a legitimate analytical concern" and defers it to Problem 9.1, but a reviewer should flag that the experiments operate outside the proven theoretical framework.

**Fix suggestion**: Either:
- (a) Provide an alternative global-flow criterion that works for $\beta_V=0$ (e.g., using the boundedness of the training distribution or the softplus regularization on $K_\theta$), or
- (b) Run one benchmark with $\beta_V > 0$ to demonstrate that the theoretical guarantees apply in at least one configuration, or
- (c) Strengthen the discussion in Remark 5.5 to explain why global flow is expected empirically (e.g., by showing that the neural network potential $V_\theta$ grows sufficiently fast on the training distribution).

### [HIGH] [Section 8, Lines 931-933] Proposition 8.4 proof is essentially empty

The proof consists of two sentences that assert "a Gronwall-type estimate gives $\|\tilde{S}_t^N u_0^N - \cS_t u_0\| \le C(N)$" without any computation. This conflates continuous and discrete flows (as noted above), does not specify the norm or the embedding/projection operator between $\R^N$ and $X$, and does not verify the Lipschitz conditions needed for Gronwall.

**Fix suggestion**: Provide a complete proof or cite a specific convergence theorem from the numerical PDE literature (e.g., from Hundsdorfer & Verwer [2003]) with the relevant stability and consistency hypotheses verified.

---

## MODERATE Issues

### [MODERATE] [Section 8, Line 948] The term $g(g^{-1}(u_0^N)) - u_0^N$ vanishes for interior data

Theorem 8.1 starts with $u_0 \in (m,M)^N$ (line 940), and $u_0^N = u_0$ since the state is already in $\R^N$. Therefore $g(g^{-1}(u_0^N)) = u_0^N$ exactly, and $\varepsilon_{\mathrm{app}}(N) = C(N)$. The bound (equation 8.4) simplifies to $C(N) + e^{L_T T}T\varepsilon_{\mathrm{match}} + C(N)$, which could be written more transparently.

If the intent is to handle initial data from the PDE space $X$ projected onto $X_N$ (where boundary values may appear), this should be stated explicitly and the boundary issue with $g^{-1}$ addressed.

**Fix suggestion**: Either simplify the bound for the stated domain $u_0 \in (m,M)^N$, or explicitly handle the PDE-to-semidiscrete projection case with appropriate boundary conditions.

### [MODERATE] [Section 8, Line 962] Missing Lipschitz verification for the semidiscrete integrator

The Gronwall argument in the transfer theorem proof (line 962-966) assumes both $\PhiT$ and $S^N$ are Lipschitz on compact subsets. While $\PhiT$'s Lipschitz property is verified in Proposition 5.7, the corresponding property for $S^N$ is not stated as an assumption or verified. Assumption 8.3 only guarantees structural preservation (invariance and dissipation), not Lipschitz continuity.

**Fix suggestion**: Add a Lipschitz condition for $S^N$ to Assumption 8.3, or verify it from the Lipschitz property of $\cF_N$.

### [MODERATE] [Section 8, Line 932] Missing Lipschitz assumption for $\cF_N$

The proof of Proposition 8.4 states "bounded by standard truncation error analysis under the Lipschitz condition on $\cF_N$" but Assumption 8.3 does not assume $\cF_N$ is Lipschitz.

**Fix suggestion**: Add Lipschitz continuity of $\cF_N$ on compact subsets of $\cK_N$ to Assumption 8.3(i).

### [MODERATE] [Section 3, Line 196] $\cE_\theta$ (learned energy) vs $\cE$ (PDE energy) not connected

Remark 3.2 defines the finite-dimensional learner with a discrete energy $\cE_N: \cK_N \to \R$, and Section 5 defines the induced discrete energy $\cE_\theta(u) = \Psi_\theta(g^{-1}(u))$. However, the relationship between $\cE_\theta$ and the discrete approximation $\cE_N$ of the PDE energy $\cE$ is never established. The dissipation guarantee (S5) at the PDE level requires $\cE(\PhiT(u,\tau)) \le \cE(u)$ with the PDE energy, while the architectural guarantee only provides $\cE_\theta(\PhiT(u,\tau)) \le \cE_\theta(u)$ with a learned surrogate.

**Fix suggestion**: Clarify in Remark 3.2 and in the discussion of Theorem 8.1 that $\cE_\theta$ is a surrogate energy that may differ from $\cE_N$, and that matching $\cE_\theta$ to $\cE_N$ is an additional requirement (related to generator matching).

### [MODERATE] [Section 7, Lines 831-834, 861] Experimental tables for Allen-Cahn and Burgers contain placeholder entries

Tables 7.4 (Allen-Cahn) and 7.6 (Burgers) contain placeholder entries denoted by bold asterisks (${\bf *}$). While this is a draft-stage issue, it means the experimental validation is incomplete—only Fisher-KPP has fully reported results.

### [MODERATE] [Section 4, Lines 270-272] Remark 4.2 identifies a gap but does not resolve it

Remark 4.2 correctly notes that the Lipschitz condition (equation 4.3) is verified only on compact subsets of the interior $(m,M)^N$ (Proposition 5.7), while Proposition 4.2 assumes it holds on the full admissible set $\cK$. The remark states "The error bound therefore applies to initial data $u_0$ whose trajectory under $\PhiT$ remains in a compact subset of the interior for $0 \le t \le T$", but this restriction is not formalized—there is no condition given under which trajectories stay away from the boundary.

### [MODERATE] [Section 5, Line 313] Domain restriction to interior set not propagated to all subsequent results

The latent construction maps $(m,M)^N \times [0,\infty) \to \cK_N$ (line 313), but the boundary states $\{u_i = m \text{ or } u_i = M\}$ are excluded. Several results (e.g., Theorem 4.1, Proposition 4.2) are stated for $u_0 \in \cK$, which includes boundary points. The boundary exclusion should be consistently stated throughout.

---

## LOW Issues

### [LOW] [Lines 71-73] Duplicate \maketitle command

`\maketitle` appears twice (lines 71 and 73). The second should be removed.

### [LOW] [Section 5, Line 451] Under-statement of regularity

The proof of Proposition 5.5 states "$g \in C^1(\R^N; (m,M)^N)$" but the sigmoid map $g$ is actually $C^\infty$. Using $C^1$ is not incorrect but is unnecessarily weak.

### [LOW] [Section 6, Lines 576-600] Proposition 6.3 is mainly a verification of definitions

Proposition 6.3 (Parameterized latent dynamics) verifies that the neural parametrization satisfies the regularity and positivity conditions of Definition 5.3. While correct, this is largely a direct check that follows from the definitions and could be condensed.

### [LOW] [Throughout] Notation $\mathrm{D}g^{-1}$ is ambiguous

The notation $\mathrm{D}g^{-1}$ could mean either $(\mathrm{D}g)^{-1}$ (inverse of the Jacobian) or $\mathrm{D}(g^{-1})$ (Jacobian of the inverse). These are equal by the inverse function theorem, but the notation should be clarified, e.g., by writing $(\mathrm{D}g)^{-1}$ or $J_{g^{-1}}$.

### [LOW] [Section 10] Open problems are well-formulated but could be more specific

The four open problems in Section 10 are clearly stated but somewhat broad. Problem 10.2 (Exact structural realization) is essentially solved by the present paper's architecture. Problem 10.4 (Continuous-to-discrete transfer) is partially addressed by Theorem 8.1. The paper could strengthen these by stating the remaining gaps more precisely.

### [LOW] [Section 7, Lines 729-730] Reproducibility: random seeds not fixed

The paper states "Random seeds are not fixed across runs." For reproducibility, it is standard practice to fix seeds or report the seed used for each run.

### [LOW] [Section 9, Line 1003] PSD completion claim needs reference or proof

Step 2 of the proof sketch claims "the partial Gram matrix specified on the edges always admits a PSD completion by choosing sufficiently large diagonal entries." While this is a known result for chordal graphs, it should reference the appropriate completion theorem (e.g., Grone et al., 1984) or note the chordality assumption.

---

## Summary of Notation Inconsistencies

1. $S_t^N$ vs $S_{\Delta t}^N$ vs $\tilde{S}_t^N$: The semidiscrete flow and its discrete approximation are not clearly distinguished (Section 8).
2. $\varepsilon_{\mathrm{app}}(T;u_0)$ (Prop 4.2) vs $\varepsilon_{\mathrm{app}}(N)$ (Theorem 8.1): Same name, different definitions.
3. $\cE$ (PDE energy, Def 3.1) vs $\cE_\theta$ (learned energy, Def 5.3) vs $\cE_N$ (discrete energy, Remark 3.2): Three different energy functionals with related but distinct roles.

---

## Overall Assessment

**Strengths:**
- Clear logical structure separating PDE specification, architectural verification, and transfer
- Correct core results: Theorem 4.1, Proposition 5.4, Corollary 5.6
- Honest acknowledgment of gaps (Remarks 5.5, 5.8, 5.9)
- Novel semigroup-level formulation for operator learning

**Weaknesses:**
- Transfer theorem (Theorem 8.1) proof has a substantive gap (dropped defect term, ambiguous flow notation)
- Universal approximation result (Proposition 9.1) is a sketch, not a proof
- Complete disconnect between theoretical assumptions ($\beta_V > 0$) and experimental configuration ($\beta_V = 0$)
- Incomplete experimental results (placeholder tables)

**Recommendation:** The paper makes a valuable conceptual contribution by formulating structure-preserving operator learning at the semigroup level. The architectural guarantees (Sections 5-6) are correctly established and the error decomposition (Section 4) is clean. However, the transfer theorem proof (the main bridge between architecture and PDE) needs significant revision, and the universal approximation result needs either a complete proof or a weakened statement. I recommend **major revision** focusing on the transfer theorem and expressivity result before the paper is suitable for a SIAM journal.
