"""
    AdaptiveDesignPointSampling(event; n = 1000, rounds = 5, round_size = 200, defensive = 0.1,
        shift = nothing, proposal = nothing, fd_step = 0.1, maxiter = 20, tol = 1.0e-3)

Importance sampling for the probability of a rare [`ViolationEvent`](@ref), with a proposal
adapted to the violation region. Needs the `ImportanceSamplers` package.

It works in standard normal space like [`DesignPointSampling`](@ref), in three stages:

 1. Find a design point `z*` for each finite bound `lo` and `hi` of `event`, with the same
    search as [`DesignPointSampling`](@ref). An event with both bounds finite is violated
    below `lo` and above `hi`. These are two separate regions, so it gets two design points.
 2. Adapt a Gaussian proposal around each design point with AMIS from ImportanceSamplers.jl,
    in `rounds` rounds of `round_size` samples. A failed power flow counts as no violation.
 3. Draw `n` fresh samples from a mixture of the adapted proposals and `N(z*, I)`, and
    weight them against that mixture. A share `defensive` of the samples comes from
    `N(z*, I)`, which keeps the weights bounded where the adapted proposal is too narrow.

Only stage 3 gives the final valid estimate. AMIS reuses samples from earlier rounds, so
its own estimate has no valid standard error. Fresh draws from the adapted proposal are
IID, so the final estimate has a valid standard error.

The supplied `event`, or its quantity, must be one of the QoIs of the problem.

```julia
using ImportanceSamplers
over = ViolationEvent(VoltageMagnitude(5), -Inf, 1.1)
result = solve(prob, AdaptiveDesignPointSampling(over; n = 2000); rng = Xoshiro(1))
violation_probability(result, over), standard_error(result, over)
```

Stages 1 and 2 can be reused for new samples by calling `solve(prob, result.method)`.
"""
struct AdaptiveDesignPointSampling{E <: ViolationEvent, P} <: AbstractPPFMethod
    event::E
    n::Int
    rounds::Int
    round_size::Int
    defensive::Float64
    shift::Union{Nothing, Vector{Vector{Float64}}}
    proposal::P
    fd_step::Float64
    maxiter::Int
    tol::Float64

    function AdaptiveDesignPointSampling(
            event::E,
            n::Integer,
            rounds::Integer,
            round_size::Integer,
            defensive::Real,
            shift::Union{Nothing, AbstractVector{<:AbstractVector{<:Real}}},
            proposal::P,
            fd_step::Real,
            maxiter::Integer,
            tol::Real,
        ) where {E <: ViolationEvent, P}
        check_positive("n", n)
        check_positive("rounds", rounds)
        check_positive("round_size", round_size)
        if !(0 <= defensive < 1)
            throw(ArgumentError("defensive must be in [0, 1), got $(defensive)"))
        end
        check_search(fd_step, maxiter, tol)
        check_reachable(event)
        return new{E, P}(
            event, n, rounds, round_size, defensive, float_vectors(shift), proposal,
            fd_step, maxiter, tol,
        )
    end
end

float_vectors(::Nothing) = nothing
float_vectors(v::AbstractVector{<:AbstractVector{<:Real}}) = [collect(Float64, s) for s in v]
