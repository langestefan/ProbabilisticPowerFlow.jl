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

See [`PowerModelsBackend`](@ref) for a fully usable example of the interface
definition.
"""
abstract type AbstractPFBackend end


"""
    init_state(backend::AbstractPFBackend, refs::AbstractVector{ComponentRef}) -> state

Allocate and return the initial mutable solver state.

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

This is a method of `CommonSolve.solve!`.
"""
CommonSolve.solve!(state, ::AbstractPFBackend)

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

"""
    PowerModelsBackend(data; alg = PowerModels.NativeNewton(), solver_kwargs = (;))
    PowerModelsBackend(filename; alg = PowerModels.NativeNewton(), solver_kwargs = (;))

An [`AbstractPFBackend`](@ref) solving the AC power flow with
[PowerModels.jl](https://github.com/lanl-ansi/PowerModels.jl).

`data` is a PowerModels network data dictionary in per unit, as returned by
`PowerModels.parse_file`. It is validated and deep copied on construction, so mutating
the dictionary afterwards does not change the backend. `filename` is a network file that
`PowerModels.parse_file` can read, such as a MATPOWER `.m` file. `alg` is the solver algorithm
handed to `PowerModels._solve_nl`, which defaults to a damped Newton method on the
analytic sparse Jacobian. Any NonlinearSolve.jl algorithm works once NonlinearSolve is
loaded.

`solver_kwargs` is a `NamedTuple` of keywords passed on to `PowerModels._solve_nl`, such as
`(abstol = 1.0e-10, maxiters = 20)` for a NonlinearSolve algorithm. `NativeNewton` takes no
keywords and is configured through its own constructor.
Only quantities `Pd` and `Qd` on loads and `Pg` on generators can be assigned. Anything
else is rejected by [`init_state`](@ref). This includes `Qd` at a PV bus, and any
assignment at the slack bus.

```julia
using ProbabilisticPowerFlow, PowerModels

backend = PowerModelsBackend("case5.m")
state = init_state(backend, [ComponentRef(ComponentField.Pd, 1)])
set_injections!(state, backend, [0.5])
info = solve!(state, backend)
extract(state, backend, VoltageMagnitude(3))
```
"""
struct PowerModelsBackend{A, K <: NamedTuple} <: AbstractPFBackend
    data::Dict{String, Any}
    alg::A
    # keywords passed on to PowerModels._solve_nl
    solver_kwargs::K
    # bus pair to branch id and whether the pair is read at the branch's from end
    branch_lookup::Dict{Tuple{Int, Int}, Tuple{String, Bool}}
    # bus pairs joined by parallel branches, for which a branch flow is ambiguous
    ambiguous_pairs::Set{Tuple{Int, Int}}
end
