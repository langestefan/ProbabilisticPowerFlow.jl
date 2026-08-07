"""
    LatinHypercube(; n = 1000, warmstart = :off, keep_inputs = false)

Latin hypercube sampling. Every germ dimension is split into `n` equal-probability
strata and each stratum is hit exactly once, with the position inside a stratum
uniform. The marginals are perfectly stratified, which reduces the variance of
smooth QoI estimates relative to plain Monte Carlo at the same `n`. The dependence
structure still comes from the copula, because the stratified uniforms enter the
same `u → germ → injections → solve` pipeline.

`warmstart` and `keep_inputs` behave as in [`MonteCarlo`](@ref).
"""
Base.@kwdef struct LatinHypercube <: AbstractPPFMethod
    n::Int = 1000
    warmstart::Symbol = :off
    keep_inputs::Bool = false
end

# One Latin hypercube in (0,1)^d as a d by n matrix. Clamped away from 0 and 1 so
# marginal quantiles of unbounded distributions stay finite.
function lhs_points(rng::AbstractRNG, d::Int, n::Int)
    U = Matrix{Float64}(undef, d, n)
    for k = 1:d
        perm = randperm(rng, n)
        for i = 1:n
            U[k, i] = (perm[i] - 1 + rand(rng)) / n
        end
    end
    return clamp!(U, eps(), 1 - eps())
end

function solve(
    prob::PPFProblem,
    method::LatinHypercube;
    rng::AbstractRNG = Random.default_rng(),
    ntasks::Integer = 1,
)
    check_warmstart(method.warmstart, prob.backend)
    U = lhs_points(rng, germ_dim(prob.model), method.n)
    return solve_u_matrix(prob, method, U; ntasks)
end
