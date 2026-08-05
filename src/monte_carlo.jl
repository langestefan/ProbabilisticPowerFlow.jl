"""
    MonteCarlo(; n = 1000, failure_policy = :record)

Plain Monte Carlo: `n` independent uniform draws through the
`u → germ → injections → solve` pipeline. `failure_policy` is `:record`.

Diverged samples land in `PPFResult.failures`.

`:retry` does a re-solve with a robust fallback solver, and is reserved for when such a
backend exists.
"""
Base.@kwdef struct MonteCarlo <: AbstractPPFMethod
    n::Int = 1000
    failure_policy::Symbol = :record
end

function solve(
    prob::PPFProblem,
    method::MonteCarlo;
    rng::AbstractRNG = Random.default_rng(),
)
    method.failure_policy == :record || throw(
        ArgumentError(
            "failure_policy $(repr(method.failure_policy)) is not implemented. " *
            "Only :record is available.",
        ),
    )
    model = prob.model
    backend = prob.backend
    d = germ_dim(model)
    n_inj = length(model.assignments)
    n_qois = length(prob.qois)

    state = init_state(backend, targets(model))

    u = Vector{Float64}(undef, d)
    germ = Vector{Float64}(undef, d)
    x = Vector{Float64}(undef, n_inj)
    samples = Matrix{Float64}(undef, n_qois, method.n)
    sample_indices = Vector{Int}(undef, method.n)
    failures = FailedSample[]
    n_converged = 0
    n_solves = 0

    for i = 1:method.n
        rand!(rng, u)
        to_physical!(x, model, u, germ)
        set_injections!(state, backend, x)
        info = solve!(state, backend)
        n_solves += 1
        if info.converged
            n_converged += 1
            for (k, q) in enumerate(prob.qois)
                samples[k, n_converged] = extract(state, backend, q)
            end
            sample_indices[n_converged] = i
        else
            push!(failures, FailedSample(i, copy(u), copy(x), info))
        end
    end

    return PPFResult(
        method,
        prob.qois,
        samples[:, 1:n_converged],
        sample_indices[1:n_converged],
        failures,
        method.n,
        n_solves,
    )
end
