"""
    MonteCarlo(; n = 1000)

Plain Monte Carlo sampling. Draws `n` independent uniform points in `(0,1)^d` and runs
one power flow per point.

Diverged samples are recorded in `failures` of the [`PPFResult`](@ref) and left out of
the statistics.

```julia
result = solve(prob, MonteCarlo(n = 2000); rng = Xoshiro(1))
```
"""
struct MonteCarlo <: AbstractPPFMethod
    n::Int

    function MonteCarlo(n::Integer)
        check_positive("n", n)
        return new(n)
    end
end

MonteCarlo(; n::Integer = 1000) = MonteCarlo(n)

function CommonSolve.solve(
        prob::PPFProblem,
        method::MonteCarlo;
        rng::AbstractRNG = Random.default_rng(),
        kwargs...,
    )
    U = rand(rng, germ_dim(prob.model), method.n)
    return solve_samples(prob, method, U; kwargs...)
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
        check_positive("n", n)
        return new(n)
    end
end

LatinHypercube(; n::Integer = 1000) = LatinHypercube(n)

function CommonSolve.solve(
        prob::PPFProblem,
        method::LatinHypercube;
        rng::AbstractRNG = Random.default_rng(),
        kwargs...,
    )
    U = lhs_points(rng, germ_dim(prob.model), method.n)
    return solve_samples(prob, method, U; kwargs...)
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
        check_positive("n", n)
        return new{S}(sampler, n)
    end
end
