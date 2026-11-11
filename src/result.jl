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
  - `weights`: `nothing` for unweighted methods, otherwise the likelihood ratio of every
    converged sample, in the column order of `samples`. See [`DesignPointSampling`](@ref).

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
    weights::Union{Nothing, Vector{Float64}}
end

PPFResult(
    method::AbstractPPFMethod,
    qois::Vector{AbstractQoI},
    samples::Matrix{Float64},
    sample_indices::Vector{Int},
    failures::Vector{FailedSample},
    n_samples::Int,
    n_solves::Int,
) = PPFResult(method, qois, samples, sample_indices, failures, n_samples, n_solves, nothing)

"""
    is_weighted(r::PPFResult)

True when the samples of `r` carry importance weights.
"""
is_weighted(r::PPFResult) = r.weights !== nothing

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

"""
    weighted_samples(r::PPFResult, q::AbstractQoI)

The samples of `q`, each multiplied by its importance weight. Their plain mean is the
importance sampling estimate of the mean of `q`.
"""
function weighted_samples(r::PPFResult, q::AbstractQoI)
    y = qoi_samples(r, q)
    if is_weighted(r)
        return y .* r.weights
    end
    return y
end

"""
    mean(r::PPFResult, q::AbstractQoI)

Estimated mean of `q`. For a weighted result this is the importance sampling estimate
`(1/n) Σ wᵢ yᵢ` over the `n` converged samples.
"""
Statistics.mean(r::PPFResult, q::AbstractQoI) = mean(weighted_samples(r, q))

function Statistics.std(r::PPFResult, q::AbstractQoI)
    check_unweighted(r, "std")
    return std(qoi_samples(r, q))
end

function Statistics.quantile(r::PPFResult, q::AbstractQoI, p)
    check_unweighted(r, "quantile")
    return quantile(qoi_samples(r, q), p)
end

function check_unweighted(r::PPFResult, name::AbstractString)
    if is_weighted(r)
        throw(
            ArgumentError(
                "$(name) is not available for a weighted result, whose samples are " *
                    "concentrated around the design point. Use mean, " *
                    "violation_probability or standard_error.",
            ),
        )
    end
    return nothing
end

"""
    violation_probability(r::PPFResult, v::ViolationEvent)

Mean estimated probability of the violation event.
"""
violation_probability(r::PPFResult, v::ViolationEvent) = mean(r, v)

"""
    standard_error(r::PPFResult, q::AbstractQoI)

Standard error of [`mean`](@ref Statistics.mean(::PPFResult, ::AbstractQoI)) of `q`, the
sample standard deviation of the (weighted) samples over `√n`. For a
[`ViolationEvent`](@ref), `standard_error(r, v) / violation_probability(r, v)` is the
coefficient of variation of the probability estimate, a common stopping criterion.

Not meaningful for quasi-Monte Carlo point sets, whose points are not independent.
"""
function standard_error(r::PPFResult, q::AbstractQoI)
    y = weighted_samples(r, q)
    return std(y) / sqrt(length(y))
end

"""
    effective_sample_size(r::PPFResult, v::ViolationEvent)

Effective number of violating samples, `(Σ wᵢ)² / Σ wᵢ²` with the sums over the converged
samples that violate `v`. For an unweighted result this is the number of violations.

This is what sets the precision of [`violation_probability`](@ref). With importance
sampling a value far below the number of violating samples means a few of them dominate
the estimate, so the shift missed part of the violation region. As a rule of thumb, an
estimate based on fewer than about 30 effective samples is not reliable.
"""
function effective_sample_size(r::PPFResult, v::ViolationEvent)
    y = weighted_samples(r, v)
    total = sum(y)
    if total == 0
        return 0.0
    end
    return total^2 / sum(abs2, y)
end
