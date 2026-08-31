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
    BranchReactivePower(from, to)

Reactive power flow on the branch `from → to`, measured at the `from` end, in per
unit.
"""
struct BranchReactivePower <: AbstractQoI
    from::Int
    to::Int
end
