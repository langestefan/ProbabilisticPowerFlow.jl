# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ProbabilisticPowerFlow.jl is a Julia package (minimum Julia 1.10) generated from
[BestieTemplate.jl](https://github.com/JuliaBesties/BestieTemplate.jl). Template-managed
files (workflows, linter configs, `.copier-answers.yml`) are updated via Copier — avoid
hand-editing `.copier-answers.yml`.

The root `Project.toml` uses the `[workspace]` feature with `test/` and `docs/` as
sub-projects, each with their own `Project.toml`.

## Architecture

The canonical design document is `archive/ppf_design.md` — read it before structural
changes. The package is a method-agnostic probabilistic power flow framework; every
sampling method uses the pipeline `u ∈ (0,1)^d → germ (copula + marginal quantiles) →
physical injections (transforms) → PF solve (backend)`.

Key seams (one file per concern in `src/`, flat includes):

- **Backend contract** (`backend_interface.jl`): `init_state` / `set_injections!` /
  `solve!` / `extract`. `solve!` returns a `SolveInfo` and must never throw on
  divergence — failures are recorded as `FailedSample` outputs, never dropped.
  `reference_backend.jl` (hand-rolled NR solver, no ecosystem deps) is the executable
  documentation of this contract; ecosystem adapters (PowerModels etc.) are planned
  as package extensions.
- **Uncertainty model** (`uncertainty.jl`, `dependence.jl`, `transforms.jl`): the
  copula always lives on the germ, never on transformed outputs. Correlation input is
  Spearman by default; `:pearson` is rejected until the Nataf correction exists —
  never silently reinterpret. Validation throws descriptive errors: PSD is checked
  after the Spearman→Gaussian mapping, and the error names the offending eigenvalue.
- **Methods** (`methods.jl`, `monte_carlo.jl`): subtypes of `AbstractPPFMethod`; they
  draw `u` and call only `to_physical!`, so samplers and dependence structures
  compose freely. `PPFResult.n_solves` counts every deterministic solve and is the
  benchmark cost currency.
- `germ_dim` is deliberately not named `dim`, which would clash with
  `Distributions.dim`.

## Commands

### Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Tests use TestItemRunner/TestItems, not plain `@testset` files. `test/runtests.jl` just
calls `@run_package_tests`; actual tests live in `test/test-*.jl` files as `@testitem`
blocks (with `@testsnippet` / `@testmodule` for shared setup, and tags like
`:unit`, `:fast`, `:integration`). To run a subset by tag:

```bash
julia --project=test -e 'using TestItemRunner; @run_package_tests filter=ti->(:unit in ti.tags)'
```

New test files must follow the `test/test-*.jl` naming pattern to be picked up.

### Linting and formatting

Pre-commit runs all linters/formatters (JuliaFormatter, markdownlint, yamllint/yamlfmt,
JSON/TOML checks) and is enforced as a git hook:

```bash
pre-commit run -a
```

Julia formatting follows `.JuliaFormatter.toml` (4-space indent, 92-char margin).
JuliaFormatter must be installed in the global Julia environment for the hook to work.

### Documentation

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); using LiveServer; servedocs()'
```

`docs/make.jl` auto-discovers pages from `docs/src/` in filename order — pages are
number-prefixed (`90-contributing.md`, `91-developer.md`, `95-reference.md`) to control
ordering. New folders under `docs/src/` require a title entry in the `titles` Dict in
`docs/make.jl`, or the build errors.

## Git workflow

- The `no-commit-to-branch` pre-commit hook blocks direct commits to `main` — always work
  on a branch.
- The repo keeps a linear history: rebase branches on `main` rather than merging.
- Branch names: dash-separated imperative, prefixed with the issue number when there is
  one (e.g. `14-add-tests`), or `typo`/`hotfix`/`small-refactor` for small no-issue
  changes.
- `CHANGELOG.md` follows Keep a Changelog: add entries under the "Unreleased" section.
