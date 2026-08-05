"""
    AbstractQoI

A scalar quantity of interest extracted from a solved power flow state via
[`extract`](@ref).
"""
abstract type AbstractQoI end

"""
    VoltageMagnitude(bus)

Voltage magnitude at `bus`, in per unit.
"""
struct VoltageMagnitude <: AbstractQoI
    bus::Int
end

"""
    VoltageAngle(bus)

Voltage angle at `bus`, in radians.
"""
struct VoltageAngle <: AbstractQoI
    bus::Int
end

"""
    BranchActivePower(from, to)

Active power flow on the branch `from → to`, measured at the `from` end, in per
unit.
"""
struct BranchActivePower <: AbstractQoI
    from::Int
    to::Int
end

"""
    ViolationEvent(qoi, lo, hi)

Indicator QoI: 1.0 when `qoi` falls outside `[lo, hi]`, 0.0 otherwise. It is a
first-class QoI rather than post-processing, so that rare-event methods such as
importance sampling and subset simulation can dispatch on it and target the event.
Its Monte Carlo mean is the violation probability.
"""
struct ViolationEvent{Q<:AbstractQoI} <: AbstractQoI
    qoi::Q
    lo::Float64
    hi::Float64
end

function extract(state, b::AbstractBackend, v::ViolationEvent)
    x = extract(state, b, v.qoi)
    return Float64(!(v.lo <= x <= v.hi))
end
