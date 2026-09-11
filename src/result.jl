"""
    FailedSample(index, u, injections, info)

A sample whose solve did not converge. It keeps its sample `index`, its point `u` in
`(0,1)^d`, the injection vector that was solved, and the [`SolveInfo`](@ref).
"""
struct FailedSample
    index::Int
    u::Vector{Float64}
    injections::Vector{Float64}
    info::SolveInfo
end

"""
    PPFResult

The output of a sampling method. Fields:

  - `method`: the PPF method that was used.
  - `qois`: the estimated quantities of interest, in the row order of `samples`.
  - `samples`: `n_qois × n_converged` matrix of QoI values.
  - `sample_indices`: column `j` of `samples` came from sample `sample_indices[j]`.
  - `failures`: the diverged samples, see [`FailedSample`](@ref).
  - `n_samples`: the total number of samples taken.
  - `n_solves`: the total number of deterministic solves attempted, failures included.

Invariant: `n_converged(r) + length(r.failures) == r.n_samples`.
"""
struct PPFResult{M <: AbstractPPFMethod}
    method::M
    qois::Vector{AbstractQoI}
    samples::Matrix{Float64}
    sample_indices::Vector{Int}
    failures::Vector{FailedSample}
    n_samples::Int
    n_solves::Int
end

"""
    n_converged(r::PPFResult)

Number of converged samples.
"""
n_converged(r::PPFResult) = size(r.samples, 2)

"""
    failure_rate(r::PPFResult)

Fraction of the samples whose solve diverged.
"""
failure_rate(r::PPFResult) = length(r.failures) / r.n_samples

"""
    qoi_index(r::PPFResult, q::AbstractQoI)

Row of QoI `q` in `r.samples`. Throws if `q` was not estimated.
"""
function qoi_index(r::PPFResult, q::AbstractQoI)
    i = findfirst(==(q), r.qois)
    if i === nothing
        throw(ArgumentError("QoI $(q) is not part of this result"))
    end
    return i
end

"""
    qoi_samples(r::PPFResult, q::AbstractQoI)

The converged samples of `q`, in the order they were sampled.
"""
qoi_samples(r::PPFResult, q::AbstractQoI) = view(r.samples, qoi_index(r, q), :)

function qoi_samples(r::PPFResult, v::ViolationEvent)
    i = findfirst(==(v), r.qois)
    if i !== nothing
        return view(r.samples, i, :)
    end

    j = findfirst(==(v.qoi), r.qois)
    if j === nothing
        throw(
            ArgumentError(
                "neither the event $(v) nor its quantity $(v.qoi) is part of this result",
            ),
        )
    end
    return [Float64(violates(v, x)) for x in view(r.samples, j, :)]
end

Statistics.mean(r::PPFResult, q::AbstractQoI) = mean(qoi_samples(r, q))
Statistics.std(r::PPFResult, q::AbstractQoI) = std(qoi_samples(r, q))
Statistics.quantile(r::PPFResult, q::AbstractQoI, p) = quantile(qoi_samples(r, q), p)

"""
    violation_probability(r::PPFResult, v::ViolationEvent)

Mean estimated probability of the violation event.
"""
violation_probability(r::PPFResult, v::ViolationEvent) = mean(r, v)
