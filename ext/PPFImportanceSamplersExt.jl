module PPFImportanceSamplersExt

using CommonSolve: CommonSolve
using ImportanceSamplers:
    ImportanceSamplers, AMIS, AMISRoundError, AllZeroWeightsError, FactorGaussian,
    LogTarget, ProposalBank, SphericalGaussian, current_proposal, importance_sample!,
    prepare_sampler, proposal_scale_factor
using LinearAlgebra: I, Symmetric, eigmin
using ProbabilisticPowerFlow:
    ProbabilisticPowerFlow, AdaptiveDesignPointSampling, PPFProblem, PPFResult,
    ViolationEvent, check_event, design_point, germ_dim, normal_cdf, solve_samples, violates
using Random: Random, AbstractRNG
using Statistics: mean

ProbabilisticPowerFlow.AdaptiveDesignPointSampling(
    event::ViolationEvent;
    n::Integer = 1000,
    rounds::Integer = 5,
    round_size::Integer = 200,
    defensive::Real = 0.1,
    shift = nothing,
    proposal = nothing,
    fd_step::Real = 0.1,
    maxiter::Integer = 20,
    tol::Real = 1.0e-3,
) = AdaptiveDesignPointSampling(
    event, n, rounds, round_size, defensive, shift, proposal, fd_step, maxiter, tol,
)

# below this smallest covariance eigenvalue an adapted proposal counts as collapsed
const COLLAPSE_EIGMIN = 1.0e-4

function CommonSolve.solve(
        prob::PPFProblem,
        method::AdaptiveDesignPointSampling;
        rng::AbstractRNG = Random.default_rng(),
        kwargs...,
    )
    check_event(prob, method.event)
    d = germ_dim(prob.model)
    events = one_sided(method.event)

    # 1. find a design point for each finite violation bound
    shift, n_search = design_points(prob, method, events, d)

    # 2. AMIS around each design point
    proposal, n_adapt = adapted_proposals(prob, method, events, shift, rng; kwargs...)
    used = AdaptiveDesignPointSampling(
        method.event, method.n, method.rounds, method.round_size, method.defensive,
        shift, proposal, method.fd_step, method.maxiter, method.tol,
    )

    # 3. IID samples from the adapted and defensive mixture
    bank = final_bank(shift, proposal, method.defensive)
    Z, logweights = draw_final(rng, bank, method.n)
    U = clamp!(normal_cdf.(Z), eps(), 1 - eps())
    r = solve_samples(prob, used, U; kwargs...)
    return PPFResult(
        used,
        r.qois,
        r.samples,
        r.sample_indices,
        r.failures,
        r.n_samples,
        r.n_solves + n_search + n_adapt,
        exp.(logweights[r.sample_indices]),
    )
end

# one event per finite bound, each violated on one side only
function one_sided(event::ViolationEvent)
    events = typeof(event)[]
    if isfinite(event.lo)
        push!(events, ViolationEvent(event.qoi, event.lo, Inf))
    end
    if isfinite(event.hi)
        push!(events, ViolationEvent(event.qoi, -Inf, event.hi))
    end
    return events
end

function design_points(prob::PPFProblem, method::AdaptiveDesignPointSampling, events, d::Int)
    if method.shift === nothing
        (; fd_step, maxiter, tol) = method
        searches = [design_point(prob, e; fd_step, maxiter, tol) for e in events]
        return [first(s) for s in searches], sum(last, searches)
    end
    check_count("shift", method.shift, events)
    for z in method.shift
        if length(z) != d
            throw(DimensionMismatch("shift has length $(length(z)), expected germ_dim $(d)"))
        end
    end
    return method.shift, 0
end

function check_count(name::AbstractString, values, events)
    if length(values) != length(events)
        throw(
            ArgumentError(
                "$(name) has $(length(values)) entries, expected one per finite bound of " *
                    "the event, $(length(events))",
            ),
        )
    end
    return nothing
end

function adapted_proposals(
        prob::PPFProblem,
        method::AdaptiveDesignPointSampling,
        events,
        shift,
        rng::AbstractRNG;
        kwargs...,
    )
    if method.proposal !== nothing
        if method.shift === nothing
            throw(ArgumentError("a fitted proposal needs the design points it was fitted at, pass `shift` too"))
        end
        check_count("proposal", method.proposal, events)
        return method.proposal, 0
    end
    d = length(first(shift))
    if method.round_size < 10 * d
        @warn "round_size = $(method.round_size) is below 10 times the $(d) germ variables, " *
            "AMIS may collapse onto a few samples"
    end
    fits = [adapt(prob, method, e, z, rng; kwargs...) for (e, z) in zip(events, shift)]
    return [first(f) for f in fits], sum(last, fits)
end

# AMIS on the indicator of `event`, started at N(z, I). Returns the proposal and the solves.
function adapt(
        prob::PPFProblem,
        method::AdaptiveDesignPointSampling,
        event::ViolationEvent,
        z::Vector{Float64},
        rng::AbstractRNG;
        kwargs...,
    )
    solves = Ref(0)
    target = indicator_target(prob, method, event, solves; kwargs...)
    alg = AMIS(SphericalGaussian(z, 1.0); rounds = method.rounds, round_size = method.round_size)
    sampler = prepare_sampler(rng, target, alg)
    try
        importance_sample!(sampler)
    catch err
        if err isa AMISRoundError && err.cause isa AllZeroWeightsError
            @warn "AMIS found no violation in round $(err.round), sampling around the " *
                "design point without adaptation"
            return unit_gaussian(z), solves[]
        end
        rethrow()
    end
    q = current_proposal(sampler)
    L = Matrix{Float64}(proposal_scale_factor(q))
    if eigmin(Symmetric(L * L')) < COLLAPSE_EIGMIN
        @warn "the adapted proposal collapsed, sampling around the design point without " *
            "adaptation. Increase round_size."
        return unit_gaussian(z), solves[]
    end
    return FactorGaussian(collect(Float64, mean(q)), L), solves[]
end

unit_gaussian(z::Vector{Float64}) = FactorGaussian(copy(z), Matrix{Float64}(I, length(z), length(z)))

logphi(z::AbstractVector) = -0.5 * (sum(abs2, z) + length(z) * log(2π))

# log φ(z) where the power flow at z converges and violates `event`, -Inf elsewhere
function indicator_target(
        prob::PPFProblem,
        method::AdaptiveDesignPointSampling,
        event::ViolationEvent,
        solves::Base.RefValue{Int};
        kwargs...,
    )
    single = PPFProblem(prob.backend, prob.model, [event.qoi])
    function batch!(values, Z)
        U = clamp!(normal_cdf.(Z), eps(), 1 - eps())
        r = solve_samples(single, method, U; kwargs...)
        solves[] += r.n_solves
        fill!(values, -Inf)
        for (j, i) in enumerate(r.sample_indices)
            if violates(event, r.samples[1, j])
                values[i] = logphi(view(Z, :, i))
            end
        end
        return nothing
    end
    function logdensity(z)
        value = [0.0]
        batch!(value, reshape(collect(Float64, z), :, 1))
        return only(value)
    end
    return LogTarget(logdensity; batch = batch!)
end

# the adapted proposals share 1 - defensive, the N(z*, I) components share defensive
function final_bank(shift, proposal, defensive::Float64)
    k = length(shift)
    components = vcat(proposal, [unit_gaussian(z) for z in shift])
    masses = vcat(fill((1 - defensive) / k, k), fill(defensive / k, k))
    return ProposalBank(components, masses)
end

# Draws samples from the bank with weights φ/q. Importance sampling estimates E_φ[f] with
# f = 1{violation}: the target is the input distribution φ, and f is applied afterwards.
# Whether a sample violates the event is known only after its power flow, so all samples
# are solved afterwards in one call.
function draw_final(rng::AbstractRNG, bank::ProposalBank, n::Int)
    function batch!(values, Z)
        for i in axes(Z, 2)
            values[i] = logphi(view(Z, :, i))
        end
        return nothing
    end
    target = LogTarget(logphi; batch = batch!)
    alg = ImportanceSamplers.ImportanceSampling(bank; nsamples = n)
    res = importance_sample!(prepare_sampler(rng, target, alg))
    return Matrix{Float64}(res.samples), Vector{Float64}(res.logweights)
end

end
