"""
    AbstractPPFMethod

A method to compute the probabilistic power flow. Sampling methods draw points
`u ∈ (0,1)^d` and only call [`to_physical!`](@ref), so any sampler works with any
dependence structure.
"""
abstract type AbstractPPFMethod end

"""
    solve(prob::PPFProblem, method::AbstractPPFMethod; rng) -> PPFResult

Estimate the QoIs of `prob` with `method`. One call runs many deterministic
[`solve!`](@ref) calls on the backend.

This is a method of `CommonSolve.solve`, the interface function shared by the SciML
ecosystem, so loading this package next to NonlinearSolve.jl gives one `solve` and no
name clash.
"""
CommonSolve.solve(::PPFProblem, ::AbstractPPFMethod)
