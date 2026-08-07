# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],
and this project adheres to [Semantic Versioning].

## [Unreleased]

- A violation band can be read off a result after the run. `violation_probability`
  and `qoi_samples` derive a `ViolationEvent` from the samples of the quantity it
  bounds, so several bands come off one run with no new solves. Declaring the
  event up front is still what lets a rare-event method target it
- A `PPFResult` displays as a summary rather than as its full sample matrix
- Nataf correction for Pearson correlation targets:
  `GaussianCopula(R, variables; correlation = :pearson)` maps a Pearson target on
  the germ to the copula parameter that induces it, given the marginals. The
  correction inverts the induced-correlation integral with a Gauss-Hermite
  product rule and a bisection, exposed on its own as `pearson_to_gaussian`, the
  Pearson counterpart of `spearman_to_gaussian`. It reproduces the closed forms
  for Gaussian, lognormal, and uniform marginals to about 1e-9, keeps zero
  targets exactly zero, and throws when a target falls outside the range skewed
  marginals can attain, naming the pair and the range. `:pearson` without
  marginals still throws, now pointing at the constructor that takes them
- `solve` and `solve!` are now methods of the CommonSolve.jl interface functions
  rather than functions this package owns. CommonSolve is the small interface
  package the SciML ecosystem shares, so `using ProbabilisticPowerFlow,
  NonlinearSolve` gives one `solve` instead of an ambiguous name that errors on
  use. Calling code is unchanged
- Pluggable solve loop for the PowerModels backend: `PowerModelsBackend` accepts
  a `solver` keyword. The default `:nlsolve` keeps the bundled path, and any
  NonlinearSolve.jl algorithm, available after `using NonlinearSolve` through the
  package extension `PPFNonlinearSolveExt`, runs the same equations on a
  nonlinear cache built once per state and reused, so repeated solves allocate
  almost nothing. On case1354 the cached path is about twice as fast serially
  and combines with `ntasks` for a 6.6 times end-to-end speedup over the old
  serial default
- The PowerModels backend bakes the net-injection arrays at the first solve and
  refreshes them per solve as a copy plus an O(slots) update. The refresh went
  from about 2 MB of Dict allocations per solve to zero, which mattered most for
  concurrent sampling
- Concurrent sampling: every sampling method accepts `ntasks` in `solve`, running
  the power flow solves on that many tasks over contiguous blocks of the solve
  order. A seed produces the same u points at every `ntasks`, with
  `warmstart = :off` the result is identical to the serial run, and warm-start
  chains run per task block. Backends must support concurrent solves on
  independent states, which `init_state` now documents
- Copulas.jl integration: any copula from Copulas.jl works directly as the germ
  dependence of an `UncertaintyModel` via the deterministic inverse Rosenblatt
  transform, so Clayton, Gumbel, Frank, t, and empirical copulas compose with
  every sampling method including QMC. The dependence seam is duck typed on
  `to_dependent!` and `dependence_dim`, and the package extension
  `PPFCopulasExt`, loaded with `using Copulas`, implements them for the foreign
  types. The compat floor is 0.1.33 because newer Copulas and PowerModels
  currently cannot share an environment through the NLSolversBase 7 versus 8
  split
- A QuasiMonteCarlo.jl sampler passes directly into `solve`, with the run
  configuration as keywords: `solve(prob, SobolSample(); n = 1024)`. Equivalent
  to wrapping it in `QMCSampling`, and the result records the full configuration
  either way
- `keep_inputs` option on every sampling method: with `keep_inputs = true` the
  u-space points of the converged samples are stored in `PPFResult.u`, aligned
  with the sample columns, so estimates can be post-processed against their
  inputs
- Latin hypercube sampling: `LatinHypercube` stratifies every germ dimension into
  `n` equal-probability strata, reducing the variance of smooth QoI estimates at
  the same sample count
- Quasi-Monte Carlo sampling: `SobolSampling` runs on a Sobol sequence with a
  random Cranley-Patterson shift, seed-reproducible and unbiased. Ships as the
  package extension `PPFSobolExt`, loaded with `using Sobol`
- QuasiMonteCarlo.jl adapter: `QMCSampling` accepts any point-set generator from
  QuasiMonteCarlo.jl, including Owen-scrambled Sobol nets, Halton, and lattice
  rules. Ships as the package extension `PPFQuasiMonteCarloExt`, loaded with
  `using QuasiMonteCarlo`
- All new methods support the same `warmstart` scheduling as `MonteCarlo`
- The PowerModels backend accepts networks with several reference buses. Each
  reference bus holds its voltage magnitude setpoint with the angle fixed at zero
  and its injection is an outcome of the solve, matching how pandapower treats
  several external grid connections
- Warm-start scheduling in `MonteCarlo`: the `warmstart` option accepts `:chain` to
  start each solve from the previous converged solution and `:sorted` to solve the
  samples in order of total injection first. Results are unchanged up to solver
  tolerance and a seed reproduces the same samples in every mode
- PowerModels.jl backend adapter: `PowerModelsBackend` builds from a PowerModels
  network data dictionary and solves with the native sparse Newton AC power flow.
  Ships as the package extension `PPFPowerModelsExt`, loaded with
  `using PowerModels`. Supports warm starts and records divergence as data. The
  solver's `PowerFlowData` is built once per state and reused across solves, so
  repeated solves skip the admittance matrix and Jacobian sparsity construction
- Initial release
- Core interface: backend contract (`init_state`, `set_injections!`, `solve!`,
  `extract`, optional `linearize`/`supports_warmstart`), quantities of interest with
  first-class `ViolationEvent`, `PPFProblem`/`PPFResult` with failures recorded as
  outputs
- Uncertainty model: germ variables with Distributions.jl marginals, assignments with
  deterministic transforms, Gaussian copula on the germ with Spearman rank
  correlation, and the `to_physical` u-space pipeline
- Dependency-free reference backend: full-Newton polar AC power flow with warm-start
  support, plus the Stagg & El-Abiad 5-bus test case (`case5`)
- Plain Monte Carlo method running the pipeline end-to-end

<!-- Links -->

[keep a changelog]: https://keepachangelog.com/en/1.1.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

<!-- Versions -->

[unreleased]: https://github.com/langestefan/ProbabilisticPowerFlow.jl/commits/main/
