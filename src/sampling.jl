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

"""
    solve_samples(prob::PPFProblem, method::AbstractPPFMethod, U::AbstractMatrix) -> PPFResult

Run one deterministic power flow per column of the `d × n` matrix `U`, where each column
is a point in `(0,1)^d`. Every solve is cold-started.

A converged sample stores its quantities of interest (QoIs). A diverged sample is kept as a
[`FailedSample`](@ref) and does not stop the loop.
"""
function solve_samples(
        prob::PPFProblem,
        method::AbstractPPFMethod,
        U::AbstractMatrix{<:Real},
    )
    (; backend, model, qois) = prob
    check_germ_rows(model, U)

    n = size(U, 2)
    state = init_state(backend, targets(model))
    x = Vector{Float64}(undef, length(model.assignments))
    germ = Vector{Float64}(undef, germ_dim(model))

    samples = Matrix{Float64}(undef, length(qois), n)
    sample_indices = Vector{Int}(undef, n)
    failures = FailedSample[]
    nc = 0

    for i in 1:n
        u = view(U, :, i)
        to_physical!(x, model, u, germ)
        set_injections!(state, backend, x)
        info = solve!(state, backend; warmstart = nothing)

        if info.converged
            nc += 1
            for (k, q) in enumerate(qois)
                samples[k, nc] = extract(state, backend, q)
            end
            sample_indices[nc] = i
        else
            push!(failures, FailedSample(i, collect(Float64, u), copy(x), info))
        end
    end

    return PPFResult(method, qois, samples[:, 1:nc], sample_indices[1:nc], failures, n, n)
end

function check_sample_count(n::Integer)
    if n < 1
        throw(ArgumentError("n must be at least 1, got $(n)"))
    end
    return nothing
end

function check_germ_rows(model::UncertaintyModel, U::AbstractMatrix)
    if size(U, 1) != germ_dim(model)
        throw(
            DimensionMismatch(
                "sample matrix has $(size(U, 1)) rows, expected germ_dim $(germ_dim(model))",
            ),
        )
    end
    return nothing
end

"""
    MonteCarlo(; n = 1000)

Plain Monte Carlo sampling. Draws `n` independent uniform points in `(0,1)^d` and runs
one cold start power flow per point.

Diverged samples are recorded in `failures` of the [`PPFResult`](@ref) and left out of
the statistics.

```julia
result = solve(prob, MonteCarlo(n = 2000); rng = Xoshiro(1))
```
"""
struct MonteCarlo <: AbstractPPFMethod
    n::Int

    function MonteCarlo(n::Integer)
        check_sample_count(n)
        return new(n)
    end
end

MonteCarlo(; n::Integer = 1000) = MonteCarlo(n)

function CommonSolve.solve(
        prob::PPFProblem,
        method::MonteCarlo;
        rng::AbstractRNG = Random.default_rng(),
    )
    U = rand(rng, germ_dim(prob.model), method.n)
    return solve_samples(prob, method, U)
end

"""
    LatinHypercube(; n = 1000)

Latin hypercube sampling. Every dimension of `(0,1)^d` is split into `n` strata of equal
probability. Each stratum is sampled once, so the `n` points are more evenly distributed
than in plain Monte Carlo sampling.

```julia
result = solve(prob, LatinHypercube(n = 500); rng = Xoshiro(1))
```
"""
struct LatinHypercube <: AbstractPPFMethod
    n::Int

    function LatinHypercube(n::Integer)
        check_sample_count(n)
        return new(n)
    end
end

LatinHypercube(; n::Integer = 1000) = LatinHypercube(n)

function CommonSolve.solve(
        prob::PPFProblem,
        method::LatinHypercube;
        rng::AbstractRNG = Random.default_rng(),
    )
    U = lhs_points(rng, germ_dim(prob.model), method.n)
    return solve_samples(prob, method, U)
end

function lhs_points(rng::AbstractRNG, d::Integer, n::Integer)
    U = Matrix{Float64}(undef, d, n)
    for k in 1:d
        strata = randperm(rng, n)
        for i in 1:n
            U[k, i] = (strata[i] - 1 + rand(rng)) / n
        end
    end
    return clamp!(U, eps(), 1 - eps())
end

"""
    QuasiMC(sampler; n = 1024)

Quasi-Monte Carlo sampling on a point set from QuasiMonteCarlo.jl.

Available samplers include `SobolSample()`, `HaltonSample()` or `LatticeRuleSample()`.

```julia
using QuasiMonteCarlo
sampler = SobolSample(R = OwenScramble(base = 2, pad = 32, rng = Xoshiro(1)))
result = solve(prob, QuasiMC(sampler; n = 1024))
```
"""
struct QuasiMC{S} <: AbstractPPFMethod
    sampler::S
    n::Int

    function QuasiMC(sampler::S, n::Integer) where {S}
        check_sample_count(n)
        return new{S}(sampler, n)
    end
end
