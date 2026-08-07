# ProbabilisticPowerFlow

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://langestefan.github.io/ProbabilisticPowerFlow.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://langestefan.github.io/ProbabilisticPowerFlow.jl/dev)
[![Test workflow status](https://github.com/langestefan/ProbabilisticPowerFlow.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/langestefan/ProbabilisticPowerFlow.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/langestefan/ProbabilisticPowerFlow.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/langestefan/ProbabilisticPowerFlow.jl)
[![Lint workflow Status](https://github.com/langestefan/ProbabilisticPowerFlow.jl/actions/workflows/Lint.yml/badge.svg?branch=main)](https://github.com/langestefan/ProbabilisticPowerFlow.jl/actions/workflows/Lint.yml?query=branch%3Amain)
[![Docs workflow Status](https://github.com/langestefan/ProbabilisticPowerFlow.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/langestefan/ProbabilisticPowerFlow.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![DOI](https://zenodo.org/badge/DOI/FIXME)](https://doi.org/FIXME)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![All Contributors](https://img.shields.io/github/all-contributors/langestefan/ProbabilisticPowerFlow.jl?labelColor=5e1ec7&color=c0ffee&style=flat-square)](#contributors)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

ProbabilisticPowerFlow.jl answers what a power flow solution looks like when the
injections are uncertain: the distribution of a voltage, the probability of a limit
violation, the tail of a loading.

One problem definition, swappable sampling methods, swappable solver backends. Every
method runs the same pipeline, so the uncertainty model, the sampler, and the power
flow solver are chosen independently:

```text
u ~ U(0,1)^d → germ (copula + marginal quantiles) → injections (transforms) → power flow
```

## Getting started

```julia
julia> using ProbabilisticPowerFlow, Distributions, Random, Statistics

julia> model = UncertaintyModel(
           # what is uncertain, with its marginal distribution
           [GermVariable("load3", Normal(1.1, 0.12)), GermVariable("load5", Normal(1.5, 0.18))],
           # where it lands in the network
           [Assignment("load3", ComponentRef(:load, "3", :pd)),
            Assignment("load5", ComponentRef(:load, "5", :pd))],
           # how the two are coupled, as Spearman rank correlation
           GaussianCopula([1.0 0.5; 0.5 1.0]),
       );

julia> undervoltage = ViolationEvent(VoltageMagnitude(5), 0.95, 1.05);

julia> prob = PPFProblem(ReferenceBackend(case5()), model, [VoltageMagnitude(5), undervoltage]);

julia> result = solve(prob, MonteCarlo(n = 2000); rng = Xoshiro(1))
PPFResult{MonteCarlo}: 2000 of 2000 samples converged in 2000 solves
  VoltageMagnitude(5)
  ViolationEvent{VoltageMagnitude}(VoltageMagnitude(5), 0.95, 1.05)

julia> mean(result, VoltageMagnitude(5)), quantile(result, VoltageMagnitude(5), 0.05)
(0.9694232767171842, 0.9544651904793491)

julia> violation_probability(result, undervoltage)
0.021
```

Diverged solves are data, never dropped: they land in `result.failures` with the
`u`-point and the injections that caused them.

## Sampling methods

Every method takes the same `warmstart`, `keep_inputs`, and `ntasks` options, so the
sampler is a one-word change.

| Method              | Notes                                                             |
| ------------------- | ----------------------------------------------------------------- |
| `MonteCarlo`        | independent draws, the reference for every other method           |
| `LatinHypercube`    | stratifies each germ dimension, lower variance at equal cost      |
| `SobolSampling`     | Sobol sequence with a random shift, unbiased and seed-reproducible |
| `QMCSampling`       | any QuasiMonteCarlo.jl point set, including Owen-scrambled nets   |

Correlation input is Spearman by default. A Pearson target runs through the Nataf
correction with `GaussianCopula(R, variables; correlation = :pearson)`.

## Backends

`ReferenceBackend` is a dependency-free Newton solver that ships with the package and
documents the backend contract. For real networks, load PowerModels.jl:

```julia
using PowerModels, NonlinearSolve

data = PowerModels.parse_file("case1354_pegase.m")
backend = PowerModelsBackend(data; solver = NewtonRaphson())
result = solve(prob, MonteCarlo(n = 1000, warmstart = :chain); ntasks = 4)
```

The solve loop is pluggable: `:nlsolve` is the bundled PowerModels path, and any
NonlinearSolve.jl algorithm reuses one nonlinear cache per state, which combined with
`ntasks` runs about 6.6 times faster end to end on a 1354-bus case.

## Extensions

Optional features load automatically with their trigger package.

| Extension       | Trigger package                                                         | Features                                        |
| --------------- | ----------------------------------------------------------------------- | ----------------------------------------------- |
| PowerModels     | [`PowerModels.jl`](https://github.com/lanl-ansi/PowerModels.jl)         | `PowerModelsBackend` for real network data      |
| NonlinearSolve  | [`NonlinearSolve.jl`](https://github.com/SciML/NonlinearSolve.jl)       | cached solve loop, `solver` keyword             |
| Copulas         | [`Copulas.jl`](https://github.com/lrnv/Copulas.jl)                      | any copula as germ dependence, tail dependence  |
| Sobol           | [`Sobol.jl`](https://github.com/stevengj/Sobol.jl)                      | `SobolSampling`                                 |
| QuasiMonteCarlo | [`QuasiMonteCarlo.jl`](https://github.com/SciML/QuasiMonteCarlo.jl)     | `QMCSampling` and direct sampler input          |

## How to Cite

If you use ProbabilisticPowerFlow.jl in your work, please cite using the reference given in [CITATION.cff](https://github.com/langestefan/ProbabilisticPowerFlow.jl/blob/main/CITATION.cff).

## Contributing

If you want to make contributions of any kind, please first that a look into our [contributing guide directly on GitHub](docs/src/contributing.md) or the [contributing page on the website](https://langestefan.github.io/ProbabilisticPowerFlow.jl/dev/contributing/)

---

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
