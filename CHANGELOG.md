# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog],
and this project adheres to [Semantic Versioning].

## [Unreleased]

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

[unreleased]: https://github.com/langestefan/ProbabilisticPowerFlow.jl/compare/v0.1.0...HEAD
