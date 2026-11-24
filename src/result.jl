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
  - `weights`: `nothing` for unweighted methods. Otherwise `weights[i]` is the likelihood
    ratio of draw `i`, failed or not. See [`DesignPointSampling`](@ref).
  - `n_setup_solves`: the solves spent before the first sample was drawn, such as a design
    point search. They are included in `n_solves`.

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
    n_setup_solves::Int
end

PPFResult(
    method::AbstractPPFMethod,
    qois::Vector{AbstractQoI},
    samples::Matrix{Float64},
    sample_indices::Vector{Int},
    failures::Vector{FailedSample},
    n_samples::Int,
    n_solves::Int,
    weights::Union{Nothing, Vector{Float64}} = nothing,
) = PPFResult(method, qois, samples, sample_indices, failures, n_samples, n_solves, weights, 0)

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

Converged samples of `q` in the order they were sampled.
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

Converged samples of `q` multiplied by their importance weight.
"""
function weighted_samples(r::PPFResult, q::AbstractQoI)
    y = qoi_samples(r, q)
    if is_weighted(r)
        return y .* view(r.weights, r.sample_indices)
    end
    return y
end

# The yᵢ taken as input by `mean` and `standard_error`. A weighted result uses wᵢ yᵢ for
# every draw and 0 for a failure.
function estimate_terms(r::PPFResult, q::AbstractQoI)
    y = weighted_samples(r, q)
    if !is_weighted(r)
        return y
    end
    terms = zeros(r.n_samples)
    terms[r.sample_indices] .= y
    return terms
end

"""
    mean(r::PPFResult, q::AbstractQoI)

Estimated mean of `q`.

An unweighted result averages `q` over the converged samples. This is the mean of `q` given
that the power flow converges.

A weighted result computes `(1/n) Σ wᵢ yᵢ` over all `n` draws, with `yᵢ = 0` for a failed
draw. For a [`ViolationEvent`](@ref) this is the probability of a violation and a converged
power flow. [`failure_probability`](@ref) is the most the failures can add.
"""
Statistics.mean(r::PPFResult, q::AbstractQoI) = mean(estimate_terms(r, q))

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

Estimated probability of the violation event, computed by
[`mean`](@ref Statistics.mean(::PPFResult, ::AbstractQoI)). A weighted result counts a
failed power flow as no violation. An unweighted result leaves it out.

The true probability lies between `p` and `p + failure_probability(r)`, up to sampling
error, where `p` is this estimate. See [`failure_probability`](@ref).
"""
violation_probability(r::PPFResult, v::ViolationEvent) = mean(r, v)

"""
    standard_error(r::PPFResult, q::AbstractQoI)

Standard error of [`mean`](@ref Statistics.mean(::PPFResult, ::AbstractQoI)) of `q`, the
standard deviation of the yᵢ taken as input by `mean`, over `√n`. For a
[`ViolationEvent`](@ref), `standard_error(r, v) / violation_probability(r, v)` is the
coefficient of variation of the probability estimate, a common stopping criterion.

Not meaningful for quasi-Monte Carlo point sets, whose points are not independent.
"""
function standard_error(r::PPFResult, q::AbstractQoI)
    y = estimate_terms(r, q)
    return std(y) / sqrt(length(y))
end

"""
    failure_probability(r::PPFResult)

Estimated probability that the power flow of a sample from the uncertainty model fails.

An unweighted result returns [`failure_rate`](@ref). A weighted result returns `(1/n) Σ wᵢ`
over the failed draws. Its draws come from the proposal, so its failure rate does not
estimate this probability.
"""
function failure_probability(r::PPFResult)
    if !is_weighted(r)
        return failure_rate(r)
    end
    return sum(f -> r.weights[f.index], r.failures; init = 0.0) / r.n_samples
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

"""
    iid_draws(method::AbstractPPFMethod)

True when `method` draws its samples independently from one distribution, so that every
prefix of the draws is itself a valid sample. Only then does [`running_estimate`](@ref)
mean anything. A new method that draws i.i.d. should add a method returning `true`.
"""
iid_draws(::AbstractPPFMethod) = false
iid_draws(::MonteCarlo) = true
iid_draws(::DesignPointSampling) = true
iid_draws(::AdaptiveDesignPointSampling) = true

"""
    running_estimate(r::PPFResult, q::AbstractQoI)
        -> (; samples, solves, estimate, standard_error)

The estimate of `q` and its standard error after each draw of `r`. Entry `i` covers the
first `i` draws. `solves[i]` counts all power flows up to that draw.

A failed draw adds `yᵢ = 0` to a weighted result and repeats the entry before it in an
unweighted one, as in [`mean`](@ref Statistics.mean(::PPFResult, ::AbstractQoI)). The last
entry equals `mean` and [`standard_error`](@ref).

Throws an `ArgumentError` for methods that do not draw independently. See
[`iid_draws`](@ref).

```julia
result = solve(prob, DesignPointSampling(event; n = 2000); rng = Xoshiro(1))
run = running_estimate(result, event)
lines(run.solves, run.estimate)
```
"""
function running_estimate(r::PPFResult, q::AbstractQoI)
    check_iid(r.method)
    y = weighted_samples(r, q)
    n = r.n_samples
    estimate = fill(NaN, n)
    se = fill(NaN, n)

    # Welford's update of the mean and the sum of squared deviations
    count = 0
    m = 0.0
    ss = 0.0
    j = 1
    for i in 1:n
        converged = j <= length(y) && r.sample_indices[j] == i
        if converged || is_weighted(r)
            term = 0.0
            if converged
                term = y[j]
                j += 1
            end
            count += 1
            delta = term - m
            m += delta / count
            ss += delta * (term - m)
        end
        if count >= 1
            estimate[i] = m
        end
        if count >= 2
            se[i] = sqrt(ss / (count - 1) / count)
        end
    end

    samples = collect(1:n)
    solves = r.n_setup_solves .+ samples
    return (; samples, solves, estimate, standard_error = se)
end

function check_iid(method::AbstractPPFMethod)
    if !iid_draws(method)
        throw(
            ArgumentError(
                "running_estimate needs independent draws, so that every prefix is a " *
                    "valid sample. $(nameof(typeof(method))) does not draw independently.",
            ),
        )
    end
    return nothing
end
