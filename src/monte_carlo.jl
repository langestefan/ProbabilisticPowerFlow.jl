"""
    MonteCarlo(; n = 1000, failure_policy = :record, warmstart = :off,
               keep_inputs = false)

Plain Monte Carlo: `n` independent uniform draws through the
`u → germ → injections → solve` pipeline.

The estimator's standard error is `σ/√n` regardless of the germ dimension, which is
what makes sampling viable at all here, and also what makes rare events expensive:
10% relative error on a violation probability `p` needs roughly `100/p` samples.

`failure_policy` is `:record`; diverged samples land in `PPFResult.failures`. With
`keep_inputs = true` the `u` points of the converged samples are stored in
`PPFResult.u`, so estimates can be post-processed against their inputs.

`warmstart` controls how consecutive solves are started, and requires a backend with
[`supports_warmstart`](@ref). The solution is the same up to solver tolerance in
every mode; only the iteration count changes.

  - `:off` solves every sample from a cold start. This is the default.
  - `:chain` solves in draw order, each solve starting from the previous converged
    solution. After a divergence the chain restarts cold.
  - `:sorted` draws all samples up front and solves them in order of total
    injection, so consecutive solves are close in injection space and the chained
    starts are better. `PPFResult.sample_indices` maps each stored column back to
    its draw index, which is no longer the storage order.
"""
Base.@kwdef struct MonteCarlo <: AbstractPPFMethod
    n::Int = 1000
    failure_policy::Symbol = :record
    warmstart::Symbol = :off
    keep_inputs::Bool = false
end

function solve(
    prob::PPFProblem,
    method::MonteCarlo;
    rng::AbstractRNG = Random.default_rng(),
    ntasks::Integer = 1,
)
    check_failure_policy(method.failure_policy)
    check_warmstart(method.warmstart, prob.backend)
    check_ntasks(ntasks)

    model = prob.model
    backend = prob.backend
    d = germ_dim(model)
    n = method.n
    u = Vector{Float64}(undef, d)

    # :sorted and concurrent solving both need the solve order to differ from the
    # draw order, so all draws happen up front. They go through the same rand! call
    # on the same buffer as the streaming path below, so a given seed produces the
    # same samples in every mode and at every ntasks.
    if method.warmstart == :sorted || ntasks > 1
        U = Matrix{Float64}(undef, d, n)
        for i = 1:n
            rand!(rng, u)
            U[:, i] = u
        end
        return solve_u_matrix(prob, method, U; ntasks)
    end

    state = init_state(backend, targets(model))
    germ = Vector{Float64}(undef, d)
    x = Vector{Float64}(undef, length(model.assignments))
    samples = Matrix{Float64}(undef, length(prob.qois), n)
    sample_indices = Vector{Int}(undef, n)
    u_kept = method.keep_inputs ? Matrix{Float64}(undef, d, n) : nothing
    failures = FailedSample[]
    converged = 0
    have_solution = false

    for i = 1:n
        rand!(rng, u)
        to_physical!(x, model, u, germ)
        set_injections!(state, backend, x)
        warm = method.warmstart == :chain && have_solution ? state : nothing
        info = solve!(state, backend; warmstart = warm)
        if info.converged
            have_solution = true
            converged += 1
            for (k, q) in enumerate(prob.qois)
                samples[k, converged] = extract(state, backend, q)
            end
            sample_indices[converged] = i
            u_kept === nothing || (u_kept[:, converged] = u)
        else
            # a diverged state holds no usable solution, so the chain restarts cold
            have_solution = false
            push!(failures, FailedSample(i, copy(u), copy(x), info))
        end
    end

    return PPFResult(
        method,
        prob.qois,
        samples[:, 1:converged],
        sample_indices[1:converged],
        failures,
        u_kept === nothing ? nothing : u_kept[:, 1:converged],
        nothing,
        n,
        n,
    )
end
