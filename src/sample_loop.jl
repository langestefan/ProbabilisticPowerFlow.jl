# Internal machinery shared by the sampling methods. Methods that draw all their
# u samples up front land in solve_u_matrix. MonteCarlo streams its draws in
# :off and :chain mode and comes here only for :sorted.

function check_failure_policy(policy::Symbol)
    policy == :record || throw(
        ArgumentError(
            "failure_policy $(repr(policy)) is not implemented. " *
            "Only :record is available.",
        ),
    )
    return nothing
end

function check_warmstart(mode::Symbol, backend::AbstractBackend)
    mode in (:off, :chain, :sorted) || throw(
        ArgumentError(
            "warmstart $(repr(mode)) is not a mode. Use :off, :chain, or :sorted.",
        ),
    )
    if mode != :off && !supports_warmstart(backend)
        throw(
            ArgumentError(
                "warmstart $(repr(mode)) requires a backend with " *
                "supports_warmstart. $(typeof(backend)) solves cold only.",
            ),
        )
    end
    return nothing
end

# Solves every column of `U` and collects the results. `method` provides the
# `warmstart` mode and is stored in the returned PPFResult. In :sorted mode the
# samples are solved in order of total injection, so consecutive solves are close
# in injection space, and `sample_indices` maps each stored column back to its
# draw index.
function solve_u_matrix(
    prob::PPFProblem,
    method::AbstractPPFMethod,
    U::AbstractMatrix{Float64},
)
    model = prob.model
    backend = prob.backend
    d = germ_dim(model)
    n = size(U, 2)
    size(U, 1) == d || throw(
        DimensionMismatch(
            "U has $(size(U, 1)) rows, expected germ dimension $(d)",
        ),
    )
    n_inj = length(model.assignments)
    n_qois = length(prob.qois)

    state = init_state(backend, targets(model))

    X = Matrix{Float64}(undef, n_inj, n)
    germ = Vector{Float64}(undef, d)
    for i = 1:n
        to_physical!(view(X, :, i), model, view(U, :, i), germ)
    end
    # Total injection is a cheap one-dimensional proxy for closeness in
    # injection space.
    order = method.warmstart == :sorted ? sortperm(vec(sum(X, dims = 1))) : (1:n)

    u = Vector{Float64}(undef, d)
    x = Vector{Float64}(undef, n_inj)
    samples = Matrix{Float64}(undef, n_qois, n)
    sample_indices = Vector{Int}(undef, n)
    u_kept = method.keep_inputs ? Matrix{Float64}(undef, d, n) : nothing
    failures = FailedSample[]
    n_converged = 0
    n_solves = 0
    have_solution = false

    for i in order
        copyto!(u, view(U, :, i))
        copyto!(x, view(X, :, i))
        set_injections!(state, backend, x)
        warm = method.warmstart != :off && have_solution ? state : nothing
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
