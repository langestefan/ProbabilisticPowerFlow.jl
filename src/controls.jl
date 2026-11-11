"""
    AbstractControl

A control device that a backend resolves inside its Newton solve instead of in an outer
loop. The [`AbstractControlModel`](@ref) in [`ControlSettings`](@ref) decides how the
piecewise control rule enters the solve.

Controls are passed to the backend, because they change the physical model and so the
estimate. See [`ReactiveLimits`](@ref) and [`SwitchedShunt`](@ref).
"""
abstract type AbstractControl end

"""
    ReactiveLimits(gen)

Holds the reactive output of generator `gen` between its `qmin` and `qmax` from the
network data.

The bus voltage is held at its setpoint while the reactive output is within its limits.
At `qmax` the voltage may only fall below the setpoint, and at `qmin` it may only rise
above it. This is the PV/PQ switching rule, solved inside Newton by the
[`AbstractControlModel`](@ref) in [`ControlSettings`](@ref).

Every active generator at the bus must be listed, since the bus has one voltage setpoint.
Their limits add up, and each generator takes the same fraction of its own range.
`qmin == qmax` is allowed and fixes the output.
"""
struct ReactiveLimits <: AbstractControl
    gen::Int
end

"""
    SwitchedShunt(shunt; steps, vset)

A shunt that switches between the discrete susceptances `steps`, in per unit, to hold
the voltage at its bus near `vset`.

The shunt is first solved as a continuous susceptance between `minimum(steps)` and
`maximum(steps)`. It holds the voltage at `vset` while it is within that range. At the
upper end the voltage may only fall below `vset`, and at the lower end it may only rise
above it. When [`ControlSettings`](@ref) has `snap = true`, the susceptance is then
snapped to the nearest step and the power flow is solved again with the shunt fixed
there.

The shunt must sit at a PQ bus. Its own `bs` in the network data is ignored. Under
[`Complementarity`](@ref) a bus can hold only one switched shunt.
"""
struct SwitchedShunt <: AbstractControl
    shunt::Int
    steps::Vector{Float64}
    vset::Float64

    function SwitchedShunt(shunt::Integer, steps::AbstractVector{<:Real}, vset::Real)
        if isempty(steps)
            throw(ArgumentError("a switched shunt needs at least one step"))
        end
        if !all(isfinite, steps)
            throw(ArgumentError("switched shunt steps must be finite, got $(steps)"))
        end
        if !(vset > 0)
            throw(ArgumentError("vset must be positive, got $(vset)"))
        end
        return new(shunt, sort!(unique(Float64.(steps))), vset)
    end
end

SwitchedShunt(shunt::Integer; steps::AbstractVector{<:Real}, vset::Real) =
    SwitchedShunt(shunt, steps, vset)

"""
    AbstractControlModel

How a backend turns the piecewise rule of each [`AbstractControl`](@ref) into equations
inside its Newton solve. See [`Complementarity`](@ref) and [`Sigmoid`](@ref).
"""
abstract type AbstractControlModel end

"""
    Complementarity(; scale = nothing)

Solves each control rule exactly, as a complementarity condition.

Each controlled bus gets one unknown `t` in place of its voltage magnitude. For a
generator with summed limits `qmin` and `qmax` it sets

```
Q = clamp(t, qmin, qmax)
V = vset - scale * (t - Q)
```

Inside the limits the voltage is at its setpoint. Beyond them the output saturates and
`t` moves the voltage instead. A switched shunt uses the same map for its susceptance.
This is Robinson's normal map, and Newton solves it as a semismooth equation. The system
keeps its size and sparsity pattern.

`scale` is the voltage change per unit of the control variable beyond its limits, in per
unit. It does not change the solution, but it decides where a Newton step that crosses a
limit lands. With `scale = nothing` each bus uses `1 / |B_ii|`, the inverse of its
self-susceptance. That approximates the voltage sensitivity of the bus, so a step that
overshoots a limit lands close to the right voltage.
"""
struct Complementarity <: AbstractControlModel
    scale::Union{Nothing, Float64}

    function Complementarity(scale::Union{Nothing, Real})
        if scale !== nothing && (!(scale > 0) || !isfinite(scale))
            throw(ArgumentError("scale must be positive and finite, got $(scale)"))
        end
        return new(scale)
    end
end

Complementarity(; scale::Union{Nothing, Real} = nothing) = Complementarity(scale)

"""
    Sigmoid(; steepness = 5000.0, homotopy = [100.0, 1000.0])

Replaces each control rule with a smooth sigmoid of the controlled voltage, following
Agarwal et al., "Continuously Differentiable Analytical Models for Implicit Control within
Power Flow", 2018. For a generator this is

```
Q(v) = qmin + (qmax - qmin) / (1 + exp(S (v - vset)))
```

`steepness` is `S`, per unit of voltage. A larger value follows the piecewise rule more
closely. At 5000 the error is below half a percent, but a steep sigmoid is hard for
Newton to converge on from a poor start.

When a solve at `steepness` fails, the backend solves again at each value of `homotopy`
in turn, each from the previous solution, before solving at `steepness` itself.
`homotopy` must be increasing and below `steepness`. An empty `homotopy` disables this.

Unlike [`Complementarity`](@ref), the result is smooth in the injections. It is also
biased, since the voltage is only near its setpoint while a limit does not bind.
"""
struct Sigmoid <: AbstractControlModel
    steepness::Float64
    homotopy::Vector{Float64}

    function Sigmoid(steepness::Real, homotopy::AbstractVector{<:Real})
        if !(steepness > 0) || !isfinite(steepness)
            throw(ArgumentError("steepness must be positive and finite, got $(steepness)"))
        end
        if !all(>(0), homotopy) || !issorted(homotopy; lt = <=)
            throw(
                ArgumentError(
                    "homotopy steepnesses must be positive and strictly increasing, got " *
                        "$(homotopy)",
                ),
            )
        end
        if !isempty(homotopy) && last(homotopy) >= steepness
            throw(
                ArgumentError(
                    "homotopy steepnesses must stay below steepness $(steepness), got " *
                        "$(homotopy)",
                ),
            )
        end
        return new(steepness, Float64.(homotopy))
    end
end

Sigmoid(; steepness::Real = 5000.0, homotopy::AbstractVector{<:Real} = [100.0, 1000.0]) =
    Sigmoid(steepness, homotopy)

"""
    ControlSettings(; model = Complementarity(), snap = true)

How a backend solves its [`AbstractControl`](@ref)s. `model` is an
[`AbstractControlModel`](@ref).

`snap = true` snaps each [`SwitchedShunt`](@ref) to its nearest step and solves again.
With `snap = false` the shunts stay continuous.
"""
struct ControlSettings
    model::AbstractControlModel
    snap::Bool
end

ControlSettings(; model::AbstractControlModel = Complementarity(), snap::Bool = true) =
    ControlSettings(model, snap)
