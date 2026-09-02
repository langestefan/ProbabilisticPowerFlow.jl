"""
    AbstractPPFMethod

A probabilistic power flow computation method.

Sampling methods differ only in how they produce `u ∈ (0,1)^d`. Everything after
that is [`to_physical!`](@ref) and the backend, so samplers and dependence
structures compose freely and a new method never touches the solve loop.

Methods are immutable and reusable. Run-specific state such as the RNG is passed to
[`solve`](@ref), not stored in the method.
"""
abstract type AbstractPPFMethod end

"""
    solve(prob::PPFProblem, method::AbstractPPFMethod; rng, ntasks = 1) -> PPFResult

Estimate the QoIs of `prob` with `method`. One framework-level `solve` runs many
deterministic backend-level [`solve!`](@ref) calls.

This is a method of `CommonSolve.solve`, the interface function the SciML ecosystem
shares, so loading this package next to NonlinearSolve.jl gives one `solve` rather
than a name clash.

With `ntasks > 1` the power flow solves run on that many concurrent tasks.
Parallelism never changes which samples are drawn: a seed produces the same `u`
points at every `ntasks`, and with `warmstart = :off` the result is identical to the
serial run. Every task solves on its own state from [`init_state`](@ref), so
backends need nothing beyond independent states. Threads must be available, for
example through `julia -t auto`, or the tasks share one thread.
"""
function solve end
