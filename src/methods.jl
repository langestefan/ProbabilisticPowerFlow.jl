"""
    AbstractPPFMethod

A probabilistic power flow computation method. Sampling methods draw `u ∈ (0,1)^d`
and call only [`to_physical!`](@ref), so samplers and dependence structures compose
freely. Methods are immutable and reusable; run-specific state (like the RNG) is
passed to [`solve`](@ref).
"""
abstract type AbstractPPFMethod end

"""
    solve(prob::PPFProblem, method::AbstractPPFMethod; kwargs...) -> PPFResult

Estimate the QoIs of `prob` with `method`. (Framework-level `solve` versus the
backend-level in-place [`solve!`](@ref): one call of `solve` triggers many
deterministic `solve!` calls.)
"""
function solve end
