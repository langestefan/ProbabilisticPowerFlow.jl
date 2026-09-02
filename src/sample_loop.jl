# The solve loop, shared by every sampling method. A method's only job is to
# produce u points; this file turns them into solves and a PPFResult. Adding a
# method must never mean writing a second loop.

function check_failure_policy(policy::Symbol)
    policy == :record || throw(
        ArgumentError(
            "failure_policy $(repr(policy)) is not implemented. Only :record is " *
            "available.",
        ),
    )
    return nothing
end

function check_warmstart(mode::Symbol, backend::AbstractPFBackend)
    mode in (:off, :chain, :sorted) || throw(
        ArgumentError(
            "warmstart $(repr(mode)) is not a mode. Use :off, :chain or :sorted.",
        ),
    )
    if mode != :off && !supports_warmstart(backend)
        throw(
            ArgumentError(
                "warmstart $(repr(mode)) requires a backend with supports_warmstart. " *
                "$(typeof(backend)) solves cold only.",
            ),
        )
    end
    return nothing
end

function check_ntasks(ntasks::Integer)
    ntasks >= 1 || throw(ArgumentError("ntasks must be at least 1, got $(ntasks)"))
    return nothing
end

# Solves every column of `U`. In :sorted mode the samples are solved in order of
# total injection, so consecutive solves are close in injection space, and
# `sample_indices` maps each stored column back to its draw index.
function solve_u_matrix(
    prob::PPFProblem,
    method::AbstractPPFMethod,
    U::AbstractMatrix{Float64};
    ntasks::Integer = 1,
)
    model = prob.model
    d = germ_dim(model)
    n = size(U, 2)
    size(U, 1) == d ||
        throw(DimensionMismatch("U has $(size(U, 1)) rows, expected germ dimension $(d)"))
    check_ntasks(ntasks)

    # the u → x mapping is cheap and deterministic, so it happens up front for
    # every sample; only the solves are worth parallelizing
    X = Matrix{Float64}(undef, length(model.assignments), n)
    germ = Vector{Float64}(undef, d)
    for i = 1:n
        to_physical!(view(X, :, i), model, view(U, :, i), germ)
    end

    # total injection is a cheap one-dimensional proxy for closeness in injection
    # space, which is what makes a warm start worth taking
    order = method.warmstart == :sorted ? sortperm(vec(sum(X, dims = 1))) : (1:n)

    ntasks == 1 && return solve_serial(prob, method, U, X, order)
    return solve_tasks(prob, method, U, X, order, ntasks)
end

function solve_serial(
    prob::PPFProblem,
    method::AbstractPPFMethod,
    U::AbstractMatrix{Float64},
    X::Matrix{Float64},
    order,
)
    backend = prob.backend
    d, n = size(U)
    n_qois = length(prob.qois)

    state = init_state(backend, targets(prob.model))
    u = Vector{Float64}(undef, d)
    x = Vector{Float64}(undef, size(X, 1))
    samples = Matrix{Float64}(undef, n_qois, n)
    sample_indices = Vector{Int}(undef, n)
    u_kept = method.keep_inputs ? Matrix{Float64}(undef, d, n) : nothing
    failures = FailedSample[]
    converged = 0
    have_solution = false

    for i in order
        copyto!(u, view(U, :, i))
        copyto!(x, view(X, :, i))
        set_injections!(state, backend, x)
        warm = method.warmstart != :off && have_solution ? state : nothing
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

# The u points are drawn before this function, so they are identical to the serial
# run, and every task works on its own state from init_state. Warm-start chains run
# inside each task's contiguous block and restart cold at block boundaries. Results
# are packed in draw-index order, which for :off makes the parallel result
# identical to the serial one.
function solve_tasks(
    prob::PPFProblem,
    method::AbstractPPFMethod,
    U::AbstractMatrix{Float64},
    X::Matrix{Float64},
    order,
    ntasks::Integer,
)
    backend = prob.backend
    model = prob.model
    d, n = size(U)
    n_qois = length(prob.qois)

    # without warm chains the chunks exist only for load balancing, so several per
    # task smooth out the cost difference between converged and diverged solves.
    # With chains, one contiguous block per task keeps the chains long.
    nchunks = method.warmstart == :off ? min(4 * ntasks, n) : min(ntasks, n)
    bounds = round.(Int, range(0, n; length = nchunks + 1))
    chunks = [order[(bounds[c]+1):bounds[c+1]] for c = 1:nchunks if bounds[c] < bounds[c+1]]

    values = Matrix{Float64}(undef, n_qois, n)
    # a plain Bool vector, not a BitVector: tasks write distinct indices
    # concurrently and a BitVector packs bits into shared words
    converged = fill(false, n)
    failures_per = [FailedSample[] for _ in eachindex(chunks)]

    tasks = map(eachindex(chunks)) do c
        Threads.@spawn begin
            state = init_state(backend, targets(model))
            u = Vector{Float64}(undef, d)
            x = Vector{Float64}(undef, size(X, 1))
            have_solution = false
            for i in chunks[c]
                copyto!(u, view(U, :, i))
                copyto!(x, view(X, :, i))
                set_injections!(state, backend, x)
                warm = method.warmstart != :off && have_solution ? state : nothing
                info = solve!(state, backend; warmstart = warm)
                if info.converged
                    have_solution = true
                    converged[i] = true
                    for (k, q) in enumerate(prob.qois)
                        values[k, i] = extract(state, backend, q)
                    end
                else
                    have_solution = false
                    push!(failures_per[c], FailedSample(i, copy(u), copy(x), info))
                end
            end
        end
    end
    foreach(wait, tasks)

    keep = findall(converged)
    failures = sort!(reduce(vcat, failures_per; init = FailedSample[]); by = f -> f.index)
    return PPFResult(
        method,
        prob.qois,
        values[:, keep],
        keep,
        failures,
        method.keep_inputs ? U[:, keep] : nothing,
        nothing,
        n,
        n,
    )
end
