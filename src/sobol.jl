"""
    SobolSampling(; n = 1000, warmstart = :off)

Quasi-Monte Carlo sampling on a Sobol low-discrepancy sequence with a random
Cranley-Patterson shift. The sequence covers the unit cube far more evenly than
independent draws, which improves the convergence rate for smooth QoIs. The shift
is drawn from the rng, so the estimate is unbiased, a seed reproduces it, and
repeated runs with different seeds give independent randomized QMC replicates.

The implementation lives in the `PPFSobolExt` package extension and needs
`using Sobol`. Without the Sobol package loaded, `solve` throws a descriptive
error.

`warmstart` behaves as in [`MonteCarlo`](@ref).
"""
Base.@kwdef struct SobolSampling <: AbstractPPFMethod
    n::Int = 1000
    warmstart::Symbol = :off
end

function solve(
    prob::PPFProblem,
    method::SobolSampling;
    rng::AbstractRNG = Random.default_rng(),
)
    ext = Base.get_extension(@__MODULE__, :PPFSobolExt)
    ext === nothing && throw(
        ArgumentError(
            "SobolSampling requires the Sobol package. Run `using Sobol` first.",
        ),
    )
    return ext.solve_sobol(prob, method, rng)
end
