# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ProbabilisticPowerFlow.jl is a Julia package (minimum Julia 1.10) generated from
[BestieTemplate.jl](https://github.com/JuliaBesties/BestieTemplate.jl). Template-managed
files (workflows, linter configs, `.copier-answers.yml`) are updated via Copier — avoid
hand-editing `.copier-answers.yml`.

The root `Project.toml` uses the `[workspace]` feature with `test/` and `docs/` as
sub-projects, each with their own `Project.toml`.

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
