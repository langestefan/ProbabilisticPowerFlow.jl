"""
    ComponentRef(kind, id, field)

Reference to a scalar quantity on a network component, e.g.
`ComponentRef(:load, "3", :pd)`. `kind` is `:load`, `:gen` or `:bus`; `id` is the
component identifier as a string (so it round-trips with the serialized uncertainty
spec); `field` names the injected quantity (`:pd`, `:qd`, `:pg`, ...).
"""
struct ComponentRef
    kind::Symbol
    id::String
    field::Symbol
end

"""
    SolveInfo(converged, iterations, residual)

Outcome of a single deterministic power flow solve. `residual` is the final
infinity-norm of the power mismatch. Backends return this from [`solve!`](@ref) and
never throw on divergence: failures are data.
"""
struct SolveInfo
    converged::Bool
    iterations::Int
    residual::Float64
end

"""
    AbstractBackend

A deterministic power flow solver backend. Required contract:

  - `init_state(backend, refs) -> state`
  - `set_injections!(state, backend, x)`
  - `solve!(state, backend; warmstart = nothing) -> SolveInfo`
  - `extract(state, backend, qoi) -> Float64`

Optional:

  - `supports_warmstart(backend) -> Bool` (defaults to `false`)
  - `linearize(backend, x0) -> (y0, S)` (unlocks cumulant/PEM methods)

See `ReferenceBackend` for an executable example of the contract. Ecosystem adapters
(PowerModels, PowerFlows, OpenDSSDirect) are added later as package extensions; the
core will declare a stub like `function PowerModelsBackend end` for the extension to
attach a method to.
"""
abstract type AbstractBackend end

"""
    init_state(backend::AbstractBackend, refs::AbstractVector{ComponentRef}) -> state

Allocate and return a mutable solver state. The backend resolves each `ComponentRef`
to an internal slot once, so that `set_injections!` is an allocation-free write per
sample. Must error loudly on a ref that does not exist in the network.
"""
function init_state end

"""
    set_injections!(state, backend::AbstractBackend, x::AbstractVector{<:Real}) -> state

Write the physical injection vector `x` into the state. `x` is ordered as the `refs`
passed to [`init_state`](@ref) (which equals the assignment order of the
`UncertaintyModel`).
"""
function set_injections! end

"""
    solve!(state, backend::AbstractBackend; warmstart = nothing) -> SolveInfo

Run a deterministic power flow solve on `state`. With `warmstart === nothing` the
backend must reset to a deterministic initial point (e.g. flat start); otherwise
`warmstart` is a previously solved state of the same backend. Must be restartable
after a failed solve, and must return a `SolveInfo` rather than throw on divergence.
"""
function solve! end

"""
    extract(state, backend::AbstractBackend, qoi::AbstractQoI) -> Float64

Read a quantity of interest from a solved state.
"""
function extract end

"""
    supports_warmstart(backend::AbstractBackend) -> Bool

Whether `solve!` accepts a previously solved state as `warmstart`. Defaults to
`false`.
"""
supports_warmstart(::AbstractBackend) = false

"""
    linearize(backend::AbstractBackend, x0) -> (y0, S)

Optional: sensitivity of the QoIs around the injection point `x0`. No methods exist
in the core yet; analytical methods (cumulant, PEM) will use it, with a
finite-difference fallback for backends that do not implement it.
"""
function linearize end
