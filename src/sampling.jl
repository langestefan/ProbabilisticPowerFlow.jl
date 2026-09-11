"""
    AbstractPPFMethod

A method to compute the probabilistic power flow. Sampling methods draw points
`u ∈ (0,1)^d` and only call [`to_physical!`](@ref), so any sampler works with any
dependence structure.
"""
abstract type AbstractPPFMethod end

"""
    solve(prob::PPFProblem, method::AbstractPPFMethod; rng, warmstart = :off, scheduler)

Estimate the QoIs of `prob` with `method`. One call runs many deterministic
[`solve!`](@ref) calls on the backend.

  - `rng`: random number generator for the samples, not used by `QuasiMC`.
  - `warmstart`: `:off` starts every solve cold, `:chain` from the previous solution, and
    `:sorted` does the same after sorting the samples by total injection.
  - `scheduler`: an OhMyThreads.jl scheduler to solve in parallel, `nothing` to solve one
    sample at a time.

This is a method of `CommonSolve.solve`, the interface function shared by the SciML
ecosystem, so loading this package next to NonlinearSolve.jl gives one `solve` and no
name clash.
"""
CommonSolve.solve(::PPFProblem, ::AbstractPPFMethod)

"""
    solve_samples(prob, method, U; warmstart = :off, scheduler = nothing) -> PPFResult

Solve one power flow per column of the `d × n` matrix `U`. Diverged samples are kept as
[`FailedSample`](@ref), and results are stored in draw order.
"""
function solve_samples(
        prob::PPFProblem,
        method::AbstractPPFMethod,
        U::AbstractMatrix{<:Real};
        warmstart::Symbol = :off,
        scheduler = nothing,
    )
    (; backend, model, qois) = prob
    check_germ_rows(model, U)
    check_warmstart(warmstart, backend)

    n = size(U, 2)
    X = physical_injections(model, U)
    values = Matrix{Float64}(undef, length(qois), n)
    converged = fill(false, n)

    chained = warmstart != :off
    failures = solve_blocks(scheduler, solve_order(X, warmstart)) do block
        solve_block!(values, converged, prob, U, X, block, chained)
    end
    sort!(failures; by = f -> f.index)

    keep = findall(converged)
    return PPFResult(method, qois, values[:, keep], keep, failures, n, n)
end

"""
    solve_blocks(f, scheduler, order) -> Vector{FailedSample}

Solve the samples in `order` by dividing into separate blocks.

`order` holds the sample indices. A block is a single slice of it. `f` solves one block and
returns the samples that failed. Every block gets its own state tracked by the backend,
so that warm starts are only re-used within the same block. The scheduler decides how to run the
blocks. If `scheduler` is `nothing`, the blocks are run serially in the current task.

Without a scheduler there is one block. Load OhMyThreads.jl to run the blocks on separate
tasks.
"""
solve_blocks(f, ::Nothing, order) = f(order)

function solve_block!(
        values::Matrix{Float64},
        converged::Vector{Bool},
        prob::PPFProblem,
        U::AbstractMatrix{<:Real},
        X::Matrix{Float64},
        block,
        chained::Bool,
    )
    (; backend, model, qois) = prob
    state = init_state(backend, targets(model))
    failures = FailedSample[]
    warm = false

    for i in block
        set_injections!(state, backend, view(X, :, i))
        if warm
            info = solve!(state, backend; warmstart = state)
        else
            info = solve!(state, backend; warmstart = nothing)
        end
        warm = chained && info.converged

        if info.converged
            for (k, q) in enumerate(qois)
                values[k, i] = extract(state, backend, q)
            end
            converged[i] = true
        else
            push!(failures, FailedSample(i, collect(Float64, view(U, :, i)), X[:, i], info))
        end
    end
    return failures
end

function physical_injections(model::UncertaintyModel, U::AbstractMatrix{<:Real})
    X = Matrix{Float64}(undef, length(model.assignments), size(U, 2))
    germ = Vector{Float64}(undef, germ_dim(model))
    for i in axes(U, 2)
        to_physical!(view(X, :, i), model, view(U, :, i), germ)
    end
    return X
end

function solve_order(X::Matrix{Float64}, warmstart::Symbol)
    if warmstart == :sorted
        return sortperm(vec(sum(X; dims = 1)))
    end
    return collect(axes(X, 2))
end

const WARMSTART_MODES = (:off, :chain, :sorted)

function check_warmstart(mode::Symbol, backend::AbstractPFBackend)
    if !(mode in WARMSTART_MODES)
        throw(ArgumentError("warmstart must be one of $(WARMSTART_MODES), got $(repr(mode))"))
    end
    if mode != :off && !supports_warmstart(backend)
        throw(
            ArgumentError(
                "warmstart $(repr(mode)) needs a backend that supports warm starts, " *
                    "$(nameof(typeof(backend))) does not",
            ),
        )
    end
    return nothing
end

function check_positive(name::AbstractString, value::Integer)
    if value < 1
        throw(ArgumentError("$(name) must be at least 1, got $(value)"))
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
