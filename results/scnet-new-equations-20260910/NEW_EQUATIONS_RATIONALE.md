# New-equation mechanism screen: rationale and attribution

The screen changes the PDE class, not optimizer settings selected after test inspection. It keeps both positive and negative results. Main candidates are nonlinear conservative diffusion and a damped sine-Gordon wave, which differ from the existing scalar reaction-diffusion and Burgers cases.

## Why these equations

Nonlinear diffusion tests reusable state-dependent transport, conservation and dissipation without a known linear heat operator doing most of the work. Learning nonlinear diffusivity from noisy data is an established inverse-PDE problem, not a new problem invented here: [Kadeethum et al., 2020](https://arxiv.org/abs/2002.08235). Thermodynamic structure in learned PDEs also has substantial prior art: [VONNs](https://www.sciencedirect.com/science/article/pii/S0022509622000692) and [OnsagerNet](https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.6.114402). Any improvements from positivity or flux form must be attributed to those interventions rather than the semigroup identity alone.

The damped sine-Gordon equation brings oscillatory propagation and a phase space (displacement, velocity). Stability theory is established: [Stability Theory for the Damped Sine-Gordon Equation](https://epubs.siam.org/doi/10.1137/0130026). Position/velocity state decomposition in learned second-order dynamics is also established: [Second Order Neural Ordinary Differential Equations](https://proceedings.neurips.cc/paper/2020/file/418db2ea5d227a9ea8db8e5357ca2084-Paper.pdf). The screen will not claim that recovering Markovian state is a novel semigroup theorem.

## Attribution

A structured autonomous generator versus an information-matched duration-conditioned generator isolates duration dependence within a shared spatial and physical representation. Both receive the same full physical state, known coefficients, training pairs and integration work. A spatially less restricted autonomous residual is a separate structural control; it does not isolate autonomy. Known-coefficient transfer is tested with frozen weights and reports limits on each coefficient, initial-state distribution and rollout interval.

No equation-specific learning-rate sweep, seed selection, loss-weight sweep or test-driven early stopping is authorized by this protocol. Numerical step sizes are chosen from stability and reference-refinement checks, not learned-model test ranking. The full protocol freezes details before formal training.
