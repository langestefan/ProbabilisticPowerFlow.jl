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
changes. The package is a method-agnostic probabilistic power flow framework. Every
sampling method uses the pipeline `u ∈ (0,1)^d → germ (copula + marginal quantiles) →
physical injections (transforms) → PF solve (backend)`.

Key seams (one file per concern in `src/`, flat includes):

- **Backend contract** (`backend_interface.jl`): `init_state` / `set_injections!` /
  `solve!` / `extract`. `solve!` returns a `SolveInfo` and must never throw on
  divergence — failures are recorded as `FailedSample` outputs, never dropped.
  `reference_backend.jl` is a hand-rolled NR solver with no ecosystem deps and serves
  as the executable documentation of this contract. Ecosystem adapters live under
  `ext/` as package extensions. Extensions cannot export new names, so the core
  module declares an empty stub function per adapter, for example
  `PowerModelsBackend`, and the extension attaches the constructor method.
  `PPFPowerModelsExt` calls the internal `PowerModels._compute_ac_pf` to get
  iterations and residual, which is why `[compat]` pins `PowerModels = "0.21"` —
  re-verify before widening. Copier-managed `TestOnPRs.yml` path filters do not
  include `ext/**`, so a PR touching only `ext/` skips PR CI. Do not hand-edit the
  workflow; the Test.yml run on merge to main covers it.
- **Uncertainty model** (`uncertainty.jl`, `dependence.jl`, `transforms.jl`): the
  copula always lives on the germ, never on transformed outputs. Correlation input is
  Spearman by default. `:pearson` is rejected until the Nataf correction exists —
  never silently reinterpret. Validation throws descriptive errors: PSD is checked
  after the Spearman→Gaussian mapping, and the error names the offending eigenvalue.
- **Methods** (`methods.jl`, `monte_carlo.jl`): subtypes of `AbstractPPFMethod`. They
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
calls `@run_package_tests`. Actual tests live in `test/test-*.jl` files as `@testitem`
blocks, with `@testsnippet` / `@testmodule` for shared setup, and tags like
`:unit`, `:fast`, `:integration`. To run a subset by tag:

```bash
cd test && julia --project=. -e 'using TestItemRunner; @run_package_tests filter=ti->(:unit in ti.tags)'
```

Run this from inside `test/`. Test discovery is cwd-sensitive, and running from the
repo root makes it scan sibling repositories in the parent folder.

New test files must follow the `test/test-*.jl` naming pattern to be picked up.

### Linting and formatting

prek runs all linters/formatters (JuliaFormatter, markdownlint, yamllint/yamlfmt,
JSON/TOML checks) and is enforced as a git hook. Install the hook once per clone
with `prek install`, then:

```bash
prek run -a
```

Julia formatting follows `.JuliaFormatter.toml` (4-space indent, 92-char margin).
JuliaFormatter must be installed in the global Julia environment for the hook to work.

### Documentation

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); using LiveServer; servedocs()'
```

`docs/make.jl` auto-discovers pages from `docs/src/` in filename order, with
`index.md` always first. The current alphabetical order (`contributing.md`,
`developer.md`, `reference.md`) is the intended order. A page that must sort
differently needs a filename chosen with that in mind. New folders under `docs/src/`
require a title entry in the `titles` Dict in `docs/make.jl`, or the build errors.

## Git workflow

- The `no-commit-to-branch` prek hook blocks direct commits to `main` — always work
  on a branch.
- The repo keeps a linear history: rebase branches on `main` rather than merging.
- Branch names: dash-separated imperative, prefixed with the issue number when there is
  one (e.g. `14-add-tests`), or `typo`/`hotfix`/`small-refactor` for small no-issue
  changes.
- `CHANGELOG.md` follows Keep a Changelog: add entries under the "Unreleased" section.
