"""
    solve_samples(prob::PPFProblem, method::AbstractPPFMethod, U::AbstractMatrix) -> PPFResult

Run one deterministic power flow per column of the `d × n` matrix `U`, where each column
is a point in `(0,1)^d`. Every solve is cold-started.

A converged sample stores its quantities of interest (QoIs). A diverged sample is kept as a
[`FailedSample`](@ref) and does not stop the loop.
"""
function solve_samples(
        prob::PPFProblem,
        method::AbstractPPFMethod,
        U::AbstractMatrix{<:Real},
    )
    (; backend, model, qois) = prob
    check_germ_rows(model, U)

    n = size(U, 2)
    state = init_state(backend, targets(model))
    x = Vector{Float64}(undef, length(model.assignments))
    germ = Vector{Float64}(undef, germ_dim(model))

    samples = Matrix{Float64}(undef, length(qois), n)
    sample_indices = Vector{Int}(undef, n)
    failures = FailedSample[]
    nc = 0

    for i in 1:n
        u = view(U, :, i)
        to_physical!(x, model, u, germ)
        set_injections!(state, backend, x)
        info = solve!(state, backend; warmstart = nothing)

        if info.converged
            nc += 1
            for (k, q) in enumerate(qois)
                samples[k, nc] = extract(state, backend, q)
            end
            sample_indices[nc] = i
        else
            push!(failures, FailedSample(i, collect(Float64, u), copy(x), info))
        end
    end

    return PPFResult(method, qois, samples[:, 1:nc], sample_indices[1:nc], failures, n, n)
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
