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
    AbstractPFBackend

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
abstract type AbstractPFBackend end


"""
    init_state(backend::AbstractPFBackend, refs::AbstractVector{ComponentRef}) -> state

Allocate and return the initial mutable solver state.

When `init_state` is called, we must look up where each `ComponentRef` is in the
backend's internal state, and keep that mapping, so that `set_injections!` can operate
as an allocation-free write for each new sample. If the ref does not exist we must throw
an error.

To enable concurrent (parallel, threaded) sampling, the backend must treat states from
separate `init_state` calls as independent, and must not mutate the backend after
construction.
"""
function init_state end

"""
    set_injections!(state, backend::AbstractPFBackend, x::AbstractVector{<:Real}) -> state

Write the physical injection vector `x` into the state. `x` is ordered according to the
`refs` passed to [`init_state`](@ref).
"""
function set_injections! end

"""
    solve!(state, backend::AbstractPFBackend; warmstart = nothing) -> SolveInfo

Run a deterministic power flow solve on `state`.

With `warmstart === nothing` the backend must reset to a deterministic initial point
such as a flat start. Otherwise `warmstart` is a previously solved state.

This is a method of `CommonSolve.solve!`, the definition here only exists to document
the interface. The backend must always implement `CommonSolve.solve!`.
"""
function solve! end

"""
    extract(state, backend::AbstractPFBackend, qoi::AbstractQoI) -> Float64

Read a quantity of interest from a solved state.
"""
function extract end

"""
    supports_warmstart(backend::AbstractPFBackend) -> Bool

Whether `solve!` accepts a previously solved state as `warmstart`. Defaults to
`false`.

This is useful for sampling methods that use a Markov chain to explore the injection
space, and can be used to accelerate convergence.
"""
supports_warmstart(::AbstractPFBackend) = false

"""
    linearize(backend::AbstractPFBackend, x0) -> (y0, S)

Optional: sensitivity around the injection point `x0`.

Some methods that rely on linearization of the power flow, such as cumulant and PEM,
will use this if implemented. Otherwise, a finite-difference fallback can be used.
"""
function linearize end
