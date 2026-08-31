# Here we describe the PPF backend interface / contract.

"""
    ComponentField

The type of a scalar quantity that can be attached to a bus. Must be one of `Pd`, `Qd`,
`Pg`, `Qg`, `Vg` or `Vm`. See [`ComponentRef`](@ref) for how these are used to refer to
a specific quantity on a specific component. The type can be accessed as
`ComponentField.T`.
"""
@enumx ComponentField Pd Qd Pg Qg Vg Vm

"""
    ComponentKind

ComponentKind is a bookkeeping enum, and can be directly derived from a 
[`ComponentField`](@ref) using [`kind`](@ref). 

Will return one of `Load`, `Gen` or `Bus`. 
"""
@enumx ComponentKind Load Gen Bus

"""
    ComponentRef(field, id)

A reference to one scalar quantity on one network component, for example
`ComponentRef(ComponentField.Pd, 3)` for the active power of load 3.

Here `field` defines the quantity, and `id` indexes the unique component. The field also 
fixes the kind of the component as given in the table below.

| field              | kind                 | `id` indexes |
|:-------------------|:---------------------|:-------------|
| `Pd`, `Qd`         | `ComponentKind.Load` | loads        |
| `Pg`, `Qg`, `Vg`   | `ComponentKind.Gen`  | generators   |
| `Vm`               | `ComponentKind.Bus`  | buses        |
"""
struct ComponentRef
    field::ComponentField.T
    id::Int
end

"""
    kind(field) -> ComponentKind.T
    kind(ref::ComponentRef) -> ComponentKind.T

Given a [`ComponentRef`](@ref), returns the ComponentKind of its field. 

Follows the table in [`ComponentRef`](@ref). Can also be used to get the kind of a field 
directly by passing a [`ComponentField`](@ref) instead of a [`ComponentRef`](@ref).

This will error if the field is not of the types defined in [`ComponentField`](@ref).
"""
function kind(f::ComponentField.T)
    if f in (ComponentField.Pd, ComponentField.Qd)
        return ComponentKind.Load
    elseif f in (ComponentField.Pg, ComponentField.Qg, ComponentField.Vg)
        return ComponentKind.Gen
    elseif f === ComponentField.Vm
        return ComponentKind.Bus
    end
    return error("no component kind is defined for $f, consider adding it to `kind`")
end

kind(ref::ComponentRef) = kind(ref.field)

"""
    SolveInfo(converged, iterations, residual)

This stores the outcome of a single deterministic power flow solve. 

`converged` is true if the solver reached the specified tolerance, `iterations` is the 
number of iterations taken by the solver, and `residual` is the final infinity-norm of 
the power mismatch. 

Backends must return this from [`solve!`](@ref) and must never throw when a solve 
diverges, leaving the state usable for the next solve.

A backend that converges must report a real `residual`. Only a diverged solve may use
`SolveInfo(false, -1, Inf)`, for a solver that reports neither the iteration count nor
the final residual.
"""
struct SolveInfo
    converged::Bool
    iterations::Int
    residual::Float64
end

"""
    AbstractPPFBackend

A deterministic power flow solver backend. Interface definition:

  - `init_state(backend, refs) -> state`
  - `set_injections!(state, backend, x)`
  - `solve!(state, backend; warmstart = nothing) -> SolveInfo`
  - `extract(state, backend, qoi) -> Float64`

Optional:

  - `supports_warmstart(backend) -> Bool`, default `false`
  - `linearize(backend, x0) -> (y0, S)`, which unlocks cumulant/PEM methods

See `ReferenceBackend` for a usable example of the contract.
"""
abstract type AbstractPPFBackend end