"""
    AbstractPPFMethod

A probabilistic power flow computation method. Sampling methods draw `u ∈ (0,1)^d`
and call only [`to_physical!`](@ref), so samplers and dependence structures compose
freely. Methods are immutable and reusable; run-specific state (like the RNG) is
passed to [`solve`](@ref).
"""
abstract type AbstractPPFMethod end

"""
    solve(prob::PPFProblem, method::AbstractPPFMethod; rng, ntasks = 1) -> PPFResult

Estimate the QoIs of `prob` with `method`. The framework-level `solve` runs many
deterministic backend-level [`solve!`](@ref) calls.

This is a method of `CommonSolve.solve`, the shared SciML interface function, so
the name is the same one NonlinearSolve.jl and the rest of that ecosystem extend
and loading them together is unambiguous.

With `ntasks > 1` the power flow solves run on that many concurrent tasks.
Parallelism never changes which samples are drawn: a seed produces the same
u points at every `ntasks`, and with `warmstart = :off` the result is identical
to the serial run. Warm-start chains run inside each task's contiguous block of
the solve order. Every task solves on its own state from [`init_state`](@ref),
so backends need nothing beyond independent states. Threads must be available,
for example through `julia -t auto`, or the tasks share one thread.
"""
function solve end
