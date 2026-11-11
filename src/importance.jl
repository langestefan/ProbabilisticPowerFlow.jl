"""
    DesignPointSampling(event; n = 1000, shift = nothing, fd_step = 0.1, maxiter = 20, tol = 1.0e-3)

Importance sampling for the probability of a rare [`ViolationEvent`](@ref).

The method works in standard normal space. Every point `z ∈ ℝ^d` maps to the unit cube by
`u = Φ(z)`, and from there to physical injections through [`to_physical!`](@ref). The
marginals, the copula and the assignment transforms of the model are used as they are.

Instead of sampling `z ~ N(0, I)`, samples are drawn from `N(z*, I)`, centered on the
design point `z*`: the most probable point at which the event just occurs. Each converged
sample `zᵢ` gets the likelihood ratio

    wᵢ = φ(zᵢ) / φ(zᵢ - z*) = exp(‖z*‖² / 2 - z*ᵀ zᵢ)

which [`violation_probability`](@ref), `mean` and [`standard_error`](@ref) apply.

The design point is found with the improved HL-RF iteration on the limit state
`g(z) = min(y - lo, hi - y)`. Here:
- `y` is the event's quantity
- `lo` and `hi` are the event's bounds, and
- `g(z) < 0` means a violation occurred.

The gradient of `g` is a forward finite difference in `z` with step `fd_step`,
which costs `d + 1` power flows per iteration, plus one or more for the line search.

See [`design_point`](@ref ProbabilisticPowerFlow.design_point) for details.

Pass `shift` to skip the search and center the samples on a user supplied point. Regardless,
the estimate will be unbiased for any shift, because `N(z*, I)` covers all of `ℝ^d`. The
shift only changes the variance.

A two-sided event, or an event that can be reached in separate regions of the input
space, gets a shift toward the nearest region only, so the other regions are sampled
poorly. Check [`standard_error`](@ref) and [`effective_sample_size`](@ref) to see how
well the shift worked.

The supplied `event`, or its quantity, must be one of the QoIs of the problem.

```julia
over = ViolationEvent(VoltageMagnitude(5), -Inf, 1.1)
result = solve(prob, DesignPointSampling(over; n = 2000); rng = Xoshiro(1))
violation_probability(result, over), standard_error(result, over)
```

You can run `solve(prob, result.method)` to sample again with the same design point,
without having to repeat the search.
"""
struct DesignPointSampling{E <: ViolationEvent} <: AbstractPPFMethod
    event::E
    n::Int
    shift::Union{Nothing, Vector{Float64}}
    fd_step::Float64
    maxiter::Int
    tol::Float64

    function DesignPointSampling(
            event::E,
            n::Integer,
            shift::Union{Nothing, AbstractVector{<:Real}},
            fd_step::Real,
            maxiter::Integer,
            tol::Real,
        ) where {E <: ViolationEvent}
        check_positive("n", n)
        check_search(fd_step, maxiter, tol)
        check_reachable(event)
        return new{E}(event, n, float_vector(shift), fd_step, maxiter, tol)
    end
end

DesignPointSampling(
    event::ViolationEvent;
    n::Integer = 1000,
    shift = nothing,
    fd_step::Real = 0.1,
    maxiter::Integer = 20,
    tol::Real = 1.0e-3,
) = DesignPointSampling(event, n, shift, fd_step, maxiter, tol)

function CommonSolve.solve(
        prob::PPFProblem,
        method::DesignPointSampling;
        rng::AbstractRNG = Random.default_rng(),
        kwargs...,
    )
    check_event(prob, method.event)
    d = germ_dim(prob.model)

    if method.shift === nothing
        (; event, fd_step, maxiter, tol) = method
        zstar, n_search = design_point(prob, event; fd_step, maxiter, tol)
    else
        if length(method.shift) != d
            throw(
                DimensionMismatch(
                    "shift has length $(length(method.shift)), expected germ_dim $(d)",
                ),
            )
        end
        zstar, n_search = method.shift, 0
    end
    used = DesignPointSampling(
        method.event, method.n, zstar, method.fd_step, method.maxiter, method.tol,
    )

    Z = randn(rng, d, method.n) .+ zstar
    U = clamp!(normal_cdf.(Z), eps(), 1 - eps())
    r = solve_samples(prob, used, U; kwargs...)

    half = 0.5 * dotp(zstar, zstar)
    weights = [exp(half - dotp(zstar, view(Z, :, i))) for i in r.sample_indices]
    return PPFResult(
        used,
        r.qois,
        r.samples,
        r.sample_indices,
        r.failures,
        r.n_samples,
        r.n_solves + n_search,
        weights,
    )
end

function check_search(fd_step::Real, maxiter::Integer, tol::Real)
    check_positive("maxiter", maxiter)
    if !(fd_step > 0) || !(tol > 0)
        throw(ArgumentError("fd_step and tol must be positive, got $(fd_step) and $(tol)"))
    end
    return nothing
end

function check_reachable(event::ViolationEvent)
    if !isfinite(event.lo) && !isfinite(event.hi)
        throw(ArgumentError("event $(event) has no finite bound, it can never occur"))
    end
    return nothing
end

float_vector(::Nothing) = nothing
float_vector(v::AbstractVector{<:Real}) = collect(Float64, v)

normal_cdf(z::Real) = cdf(Normal(), z)

dotp(a::AbstractVector, b::AbstractVector) = sum(a[i] * b[i] for i in eachindex(a, b))

norm2(v::AbstractVector) = sqrt(dotp(v, v))

function check_event(prob::PPFProblem, event::ViolationEvent)
    if !(event in prob.qois) && !(event.qoi in prob.qois)
        throw(
            ArgumentError(
                "neither the event $(event) nor its quantity $(event.qoi) is a QoI of " *
                    "the problem, so its probability could not be read from the result",
            ),
        )
    end
    return nothing
end

"""
    LimitState(prob, event)

Evaluates the limit state `g(z) = min(y - lo, hi - y)` of `event` at points `z` in
standard normal space, one cold-started power flow per call. A diverged solve gives
`NaN`. `n_solves` counts the calls.
"""
mutable struct LimitState{P <: PPFProblem, E <: ViolationEvent, S}
    const prob::P
    const event::E
    const state::S
    const x::Vector{Float64}
    const germ::Vector{Float64}
    const u::Vector{Float64}
    n_solves::Int
end

function LimitState(prob::PPFProblem, event::ViolationEvent)
    (; backend, model) = prob
    state = init_state(backend, targets(model))
    x = Vector{Float64}(undef, length(model.assignments))
    d = germ_dim(model)
    return LimitState(prob, event, state, x, Vector{Float64}(undef, d), zeros(d), 0)
end

function (g::LimitState)(z::AbstractVector{<:Real})
    (; backend, model) = g.prob
    g.u .= clamp.(normal_cdf.(z), eps(), 1 - eps())
    to_physical!(g.x, model, g.u, g.germ)
    set_injections!(g.state, backend, g.x)
    g.n_solves += 1
    info = solve!(g.state, backend; warmstart = nothing)
    if !info.converged
        return NaN
    end
    y = extract(g.state, backend, g.event.qoi)
    return min(y - g.event.lo, g.event.hi - y)
end

"""
    limit_state_gradient(g, z, gz, h) -> Vector{Float64}

Forward finite difference gradient of `g` at `z`, where `gz = g(z)`. A coordinate whose
forward point diverges falls back to a backward difference.
"""
function limit_state_gradient(g::LimitState, z::Vector{Float64}, gz::Float64, h::Float64)
    grad = similar(z)
    zh = copy(z)
    for k in eachindex(z)
        zh[k] = z[k] + h
        forward = g(zh)
        if isnan(forward)
            zh[k] = z[k] - h
            backward = g(zh)
            if isnan(backward)
                throw(
                    ErrorException(
                        "power flow diverged on both sides of germ variable $(k) during " *
                            "the design point search, pass a `shift` instead",
                    ),
                )
            end
            grad[k] = (gz - backward) / h
        else
            grad[k] = (forward - gz) / h
        end
        zh[k] = z[k]
    end
    return grad
end

"""
    design_point(prob, event; fd_step = 0.1, maxiter = 20, tol = 1.0e-3) -> (z*, n_solves)

Search for the most probable point of the limit state surface `g(z) = 0` of `event` with
the improved HL-RF method (Zhang and Der Kiureghian, 1995). The gradient is a forward
finite difference with step `fd_step`.

Each iteration linearizes `g` at `z` and finds the point of the linearized surface that is
closest to the origin:

    z̃ = (∇gᵀ z - g(z)) / ‖∇g‖² · ∇g

Plain HL-RF jumps to `z̃` and can oscillate when `g` is strongly curved, as it is with
light-tailed marginals or an asymmetric copula. Here the step `z → z̃` is halved until the
merit function `½‖z‖² + c |g(z)|` decreases, with `c = 2 max(‖z‖, ‖z̃‖) / ‖∇g‖`, and also
while the power flow diverges, at most 10 times. The search has converged when the full
step `‖z̃ - z‖` is below `tol · max(1, ‖z‖)`.
"""
function design_point(
        prob::PPFProblem,
        event::ViolationEvent;
        fd_step::Real = 0.1,
        maxiter::Integer = 20,
        tol::Real = 1.0e-3,
    )
    g = LimitState(prob, event)
    h = Float64(fd_step)
    z = zeros(germ_dim(prob.model))
    gz = g(z)
    if isnan(gz)
        throw(ErrorException("the power flow diverged at the median injections"))
    end

    for _ in 1:maxiter
        grad = limit_state_gradient(g, z, gz, h)
        nrm = norm2(grad)
        if nrm == 0
            throw(
                ErrorException(
                    "the event quantity does not respond to any germ variable at " *
                        "z = $(z), increase `fd_step` or pass a `shift`",
                ),
            )
        end
        target = ((dotp(grad, z) - gz) / nrm^2) .* grad
        step = target .- z
        if norm2(step) <= tol * max(1.0, norm2(z))
            return z, g.n_solves
        end

        c = 2 * max(norm2(z), norm2(target)) / nrm
        merit(v, gv) = 0.5 * dotp(v, v) + c * abs(gv)
        m0 = merit(z, gz)
        znew = z .+ step
        gnew = g(znew)
        halvings = 0
        while (isnan(gnew) || merit(znew, gnew) >= m0) && halvings < 10
            step ./= 2
            znew = z .+ step
            gnew = g(znew)
            halvings += 1
        end
        if isnan(gnew)
            throw(ErrorException("the power flow diverged along the design point search"))
        end
        z, gz = znew, gnew
    end

    @warn "the design point search did not converge in $(maxiter) iterations, " *
        "sampling around the last point. The estimate stays unbiased, but its variance " *
        "may be larger."
    return z, g.n_solves
end
