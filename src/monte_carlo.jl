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
        if n < 1
            throw(ArgumentError("n must be at least 1, got $(n)"))
        end
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
