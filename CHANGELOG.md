# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],
and this project adheres to [Semantic Versioning].

## [Unreleased]

### Added

- Backend interface: `ComponentRef`, `SolveInfo`, `AbstractPFBackend` with `init_state`,
  `set_injections!`, `solve!` and `extract`, plus the optional `supports_warmstart` and
  `linearize`.
- `PowerModelsBackend`, an AC power flow backend on PowerModels.jl, loaded as a package
  extension.
- `solver_kwargs` on `PowerModelsBackend`, passing keywords such as `abstol` and
  `maxiters` on to a NonlinearSolve.jl algorithm.
- A NonlinearSolve.jl algorithm on `PowerModelsBackend` keeps one solver cache per state
  and reuses it for every solve.
- Quantities of interest: `VoltageMagnitude`, `VoltageAngle`, `BranchActivePower`,
  `BranchReactivePower` and `ViolationEvent`.
- `UncertaintyModel` from germ variables, assignments and a Copulas.jl copula, with
  `IdentityTransform` and `AffineTransform`.
- `PPFProblem`, `solve` as a method of `CommonSolve.solve`, and `MonteCarlo` sampling.
- `LatinHypercube` sampling, and `QuasiMC` for any QuasiMonteCarlo.jl sampler, loaded
  as a package extension.
- Warm starts with `warmstart = :chain` or `:sorted`, and parallel solves with an
  OhMyThreads.jl `scheduler`, as keywords of `solve`.
- `PPFResult` with diverged samples kept as `FailedSample`, and the statistics
  `mean`, `std`, `quantile` and `violation_probability`.
- Compact one-line and tree displays for models, problems, results and backends.
- Performance benchmarks in `benchmark/`, run with AirspeedVelocity.jl locally and on
  pull requests.

<!-- Links -->

[keep a changelog]: https://keepachangelog.com/en/1.1.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

<!-- Versions -->

[unreleased]: https://github.com/langestefan/ProbabilisticPowerFlow.jl/compare/v0.1.0...HEAD
