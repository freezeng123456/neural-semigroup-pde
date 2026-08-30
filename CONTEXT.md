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
