# Neural PDE Flow Context

This glossary fixes the vocabulary used to separate spatial admissibility from
temporal composition in the repository's neural PDE experiments.

## Current main line

The central object is a **boundary-admissible neural semigroup**. For a
discrete spatial boundary operator `B_h`, define

\[
X_B=\{u:B_hu=g\}.
\]

A hard boundary map places the state in `X_B`, and a tangent vector-field map
keeps every integration stage in `X_B`. A duration-independent vector field
then defines one autonomous flow on that space. These two interventions have
different responsibilities:

\[
\text{boundary map/tangent generator}\Rightarrow S_t(X_B)\subseteq X_B,
\qquad
\text{autonomy}\Rightarrow S_{t+s}=S_t\circ S_s
\]

for the exact continuous flow, with a separately measured numerical
composition defect for finite-step integration.

Wave 2 implements this construction for discrete nonhomogeneous Dirichlet,
homogeneous Neumann, and Robin conditions. The boundary-family result is an
implementation and mechanism result, not a claim that the semigroup axiom
itself selects a boundary condition. Boundary residual, temporal composition
defect, and PDE prediction error remain independent endpoints.

## Guarantee hierarchy

Use the following implication chain and do not skip a level:

\[
\text{affine boundary retraction + tangent generator}
\Longrightarrow
\text{fixed-grid boundary invariance},
\]

\[
\text{autonomous globally Lipschitz finite-dimensional generator}
\Longrightarrow
\text{unique global forward semiflow and exact composition},
\]

\[
\text{PDE-specific spatial consistency + generator matching + stability}
\Longrightarrow
\text{conditional PDE rollout accuracy}.
\]

The first two implications are proved for the current hard autonomous
boundary-family Tanh model in boundary_admissible_semigroup_theory.tex. The
third is a conditional theorem whose hypotheses must be completed separately
for every PDE using docs/research/PDE_THEOREM_CARD.md.

The current boundary-family model does not structurally guarantee a pointwise
maximum principle, mass conservation, physical-energy decay, or a
mesh-uniform one-sided stability constant. A small composition defect can
therefore coexist with a small, large, or biased PDE prediction error.

For an explicitly time-dependent PDE, use an evolution family \(U(t,s)\) with
\(U(t,r)U(r,s)=U(t,s)\), or augment the state with a clock and redefine the
problem. A requested duration supplied to a query-time model is not by itself
an absolute clock.

## Language

**Spatial boundary condition**:
A constraint on the trace or flux of a PDE state at the spatial boundary, such
as homogeneous Dirichlet, Neumann, or Robin data.
_Avoid_: State bound, semigroup boundary

**Boundary-admissible state space**:
The set of discrete states that satisfy the selected spatial boundary
condition. A boundary-compatible flow maps this set into itself at every time.
_Avoid_: Valid range, bounded state space

**Hard boundary compatibility**:
A parameter-free structural guarantee that every model stage remains in the
boundary-admissible state space, independent of training data or loss values.
_Avoid_: Boundary penalty, learned boundary fit

**State bound**:
A pointwise range constraint such as $0\leq u\leq1$ throughout the spatial
domain. It is not a spatial boundary condition.
_Avoid_: Dirichlet boundary, boundary constraint

**Temporal semigroup**:
A time-homogeneous flow family satisfying
$S_{t+s}=S_t\circ S_s$ on its state space.
_Avoid_: Boundary semigroup, rollout consistency alone

**Physical invariant**:
A quantity that is conserved or monotone along a PDE flow, such as mass or
free energy. It is distinct from both a boundary condition and a state bound.
_Avoid_: Boundary condition, semigroup law

**Numerical composition defect**:
The discrepancy between two finite-step integration paths with the same total
time. For autonomous RK4 it converges to zero at fourth order under the stated
smoothness and stability hypotheses, but it is neither generator error nor PDE
prediction error.
_Avoid_: Exact semigroup proof, transfer certificate

**Evolution family**:
The two-time object \(U(t,s)\) for explicitly nonautonomous dynamics, satisfying
\(U(t,r)U(r,s)=U(t,s)\). It should not be replaced by a one-parameter
semigroup unless the state has been correctly augmented.
_Avoid_: Query-duration semigroup
