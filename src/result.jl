"""
    FailedSample(index, u, injections, info)

A sample whose deterministic solve did not converge, recorded with its original
sample `index`, its `u`-space point, and the physical injection vector that caused
the failure. Failures are outputs, never silently dropped: divergence is information
about the feasibility boundary.
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
    only — statistics never accidentally include failures.
  - `sample_indices`: column `j` of `samples` came from original sample
    `sample_indices[j]`, keeping every sample traceable to its `u`-point.
  - `failures`: the diverged samples, see [`FailedSample`](@ref).
  - `u`: the `(d × n_converged)` matrix of u-space points of the converged
    samples, column `j` matching column `j` of `samples`. Only kept when the
    method was built with `keep_inputs = true`, otherwise `nothing`. Failed
    samples always carry their u-point in `failures`.
  - `n_samples`: the requested sample budget.
  - `n_solves`: number of deterministic power flow solve calls, including failed
    ones — the hardware-independent cost currency of the benchmark protocol.

Invariant: `n_converged(result) + length(result.failures) == result.n_samples`.
"""
struct PPFResult{M<:AbstractPPFMethod}
    method::M
    qois::Vector{AbstractQoI}
    samples::Matrix{Float64}
    sample_indices::Vector{Int}
    failures::Vector{FailedSample}
    u::Union{Nothing,Matrix{Float64}}
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

Row index of QoI `q` in `r.samples`; errors if `q` was not estimated.
"""
function qoi_index(r::PPFResult, q::AbstractQoI)
    i = findfirst(==(q), r.qois)
    i === nothing && throw(ArgumentError("QoI $(q) is not part of this result"))
    return i
end

qoi_samples(r::PPFResult, q::AbstractQoI) = view(r.samples, qoi_index(r, q), :)

Statistics.mean(r::PPFResult, q::AbstractQoI) = mean(qoi_samples(r, q))
Statistics.std(r::PPFResult, q::AbstractQoI) = std(qoi_samples(r, q))
Statistics.quantile(r::PPFResult, q::AbstractQoI, p) = quantile(qoi_samples(r, q), p)

"""
    violation_probability(r::PPFResult, v::ViolationEvent)

Estimated probability of the violation event: the mean of its indicator samples.
"""
violation_probability(r::PPFResult, v::ViolationEvent) = mean(r, v)
