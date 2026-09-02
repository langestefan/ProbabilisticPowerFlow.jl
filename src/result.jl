"""
    FailedSample(index, u, injections, info)

A sample whose deterministic solve did not converge, recorded with its draw
`index`, its `u`-space point and the injection vector that caused the failure.

Failures are outputs, never silently dropped: divergence is information about the
feasibility boundary, and a run that quietly discards it reports a violation
probability computed on the wrong population.
"""
struct FailedSample
    index::Int
    u::Vector{Float64}
    injections::Vector{Float64}
    info::SolveInfo
end

"""
    PPFResult

Output of a sampling method. Fields:

  - `method`: the method that produced the result.
  - `qois`: the estimated quantities of interest, in the row order of `samples`.
  - `samples`: `(n_qois × n_converged)` matrix of QoI values, converged samples
    only, so statistics never accidentally include failures.
  - `sample_indices`: column `j` of `samples` came from draw `sample_indices[j]`,
    which keeps every sample traceable to its `u` point.
  - `failures`: the diverged samples, see [`FailedSample`](@ref).
  - `u`: the `(d × n_converged)` matrix of `u`-space points, column `j` matching
    column `j` of `samples`. Only kept when the method was built with
    `keep_inputs = true`, otherwise `nothing`. Failed samples always carry their
    `u` point in `failures`.
  - `weights`: per-sample likelihood ratios, or `nothing` when the samples are
    drawn from the model's own distribution. Plain Monte Carlo is unweighted;
    importance sampling is not.
  - `n_samples`: the requested sample budget.
  - `n_solves`: number of deterministic solve calls, failures included. This is the
    hardware-independent cost of a run, and the currency methods are compared in.

Invariant: `n_converged(result) + length(result.failures) == result.n_samples`.
"""
struct PPFResult{M<:AbstractPPFMethod}
    method::M
    qois::Vector{AbstractQoI}
    samples::Matrix{Float64}
    sample_indices::Vector{Int}
    failures::Vector{FailedSample}
    u::Union{Nothing,Matrix{Float64}}
    weights::Union{Nothing,Vector{Float64}}
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

Fraction of the sample budget whose solves diverged.
"""
failure_rate(r::PPFResult) = length(r.failures) / r.n_samples

"""
    qoi_index(r::PPFResult, q::AbstractQoI)

Row index of `q` in `r.samples`. Errors if `q` was not estimated.
"""
function qoi_index(r::PPFResult, q::AbstractQoI)
    i = findfirst(==(q), r.qois)
    i === nothing && throw(ArgumentError("QoI $(q) is not part of this result"))
    return i
end

"""
    qoi_samples(r::PPFResult, q::AbstractQoI)

The converged samples of `q`, in storage order.

A [`ViolationEvent`](@ref) that was not itself estimated is derived from the samples
of its own quantity, because the indicator is a deterministic function of that
quantity. Any band on a recorded quantity can therefore be evaluated after the run,
with no new solves.
"""
qoi_samples(r::PPFResult, q::AbstractQoI) = view(r.samples, qoi_index(r, q), :)

function qoi_samples(r::PPFResult, v::ViolationEvent)
    i = findfirst(==(v), r.qois)
    i === nothing || return view(r.samples, i, :)
    j = findfirst(==(v.qoi), r.qois)
    j === nothing && throw(
        ArgumentError(
            "neither the event $(v) nor its quantity $(v.qoi) is part of this result",
        ),
    )
    return [Float64(!(v.lo <= x <= v.hi)) for x in view(r.samples, j, :)]
end

# Unweighted results take the plain Statistics path. Weighted ones are self-
# normalized: dividing by the weight sum rather than by n keeps the estimator
# unbiased when the weights do not average to one.
Statistics.mean(r::PPFResult, q::AbstractQoI) = _wmean(qoi_samples(r, q), r.weights)

_wmean(x, ::Nothing) = mean(x)
_wmean(x, w) = sum(w[k] * x[k] for k in eachindex(x)) / sum(w)

function Statistics.std(r::PPFResult, q::AbstractQoI)
    x = qoi_samples(r, q)
    r.weights === nothing && return std(x)
    mu = _wmean(x, r.weights)
    return sqrt(_wmean((x .- mu) .^ 2, r.weights))
end

Statistics.quantile(r::PPFResult, q::AbstractQoI, p) = quantile(qoi_samples(r, q), p)

"""
    violation_probability(r::PPFResult, v::ViolationEvent)

Estimated probability of the violation event: the mean of its indicator samples.

The event does not have to have been part of the run, as long as the quantity it
bounds was, so several bands can be read off one result.
"""
violation_probability(r::PPFResult, v::ViolationEvent) = mean(r, v)
