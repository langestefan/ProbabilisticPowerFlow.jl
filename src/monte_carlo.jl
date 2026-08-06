"""
    MonteCarlo(; n = 1000, failure_policy = :record, warmstart = :off,
               keep_inputs = false)

Plain Monte Carlo: `n` independent uniform draws through the
`u → germ → injections → solve` pipeline. `failure_policy` is `:record`.

Diverged samples land in `PPFResult.failures`. With `keep_inputs = true` the
u-space points of the converged samples are stored in `PPFResult.u`, so estimates
can be post-processed against their inputs.

`:retry` does a re-solve with a robust fallback solver, and is reserved for when such a
backend exists.

`warmstart` controls how consecutive solves are started. It requires a backend with
`supports_warmstart`. The solution is the same up to solver tolerance in every mode.
Only the iteration count changes.

  - `:off` solves every sample from a cold start. This is the default.
  - `:chain` solves the samples in draw order and starts each solve from the
    previous converged solution. After a diverged sample the chain restarts cold.
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
)
    check_failure_policy(method.failure_policy)
    check_warmstart(method.warmstart, prob.backend)

    model = prob.model
    backend = prob.backend
    d = germ_dim(model)
    n_inj = length(model.assignments)
    n_qois = length(prob.qois)
    n = method.n

    u = Vector{Float64}(undef, d)

    # In :sorted mode all draws happen up front so the solve order can differ
    # from the draw order. The draws go through the same rand! call on the same
    # buffer as the streaming modes, so a given seed produces the same samples in
    # every mode.
    if method.warmstart == :sorted
        U = Matrix{Float64}(undef, d, n)
        for i = 1:n
            rand!(rng, u)
            U[:, i] = u
        end
        return solve_u_matrix(prob, method, U)
    end

    state = init_state(backend, targets(model))

    germ = Vector{Float64}(undef, d)
    x = Vector{Float64}(undef, n_inj)
    samples = Matrix{Float64}(undef, n_qois, n)
    sample_indices = Vector{Int}(undef, n)
    u_kept = method.keep_inputs ? Matrix{Float64}(undef, d, n) : nothing
    failures = FailedSample[]
    n_converged = 0
    n_solves = 0
    have_solution = false

    for i = 1:n
        rand!(rng, u)
        to_physical!(x, model, u, germ)
        set_injections!(state, backend, x)
        warm = method.warmstart == :chain && have_solution ? state : nothing
        info = solve!(state, backend; warmstart = warm)
        n_solves += 1
        if info.converged
            have_solution = true
            n_converged += 1
            for (k, q) in enumerate(prob.qois)
                samples[k, n_converged] = extract(state, backend, q)
            end
            sample_indices[n_converged] = i
            u_kept === nothing || (u_kept[:, n_converged] = u)
        else
            # A diverged state holds no usable solution, so the chain restarts
            # from a cold start.
            have_solution = false
            push!(failures, FailedSample(i, copy(u), copy(x), info))
        end
    end

    return PPFResult(
        method,
        prob.qois,
        samples[:, 1:n_converged],
        sample_indices[1:n_converged],
        failures,
        u_kept === nothing ? nothing : u_kept[:, 1:n_converged],
        n,
        n_solves,
    )
end
