"""
    Branch(f, t, y, b_half)

A π-model branch from bus `f` to bus `t` with series admittance `y` and half of the
total line-charging susceptance `b_half` at each end.
"""
struct Branch
    f::Int
    t::Int
    y::ComplexF64
    b_half::Float64
end

Branch(f::Int, t::Int, r::Real, x::Real, b_total::Real) =
    Branch(f, t, inv(complex(r, x)), b_total / 2)

"""
    build_ybus(n, branches)

Dense bus admittance matrix of the π-model branches (kept as a separate pure
function so tests can spot-check entries independently of the solver).
"""
function build_ybus(n::Int, branches::AbstractVector{Branch})
    Y = zeros(ComplexF64, n, n)
    for br in branches
        Y[br.f, br.f] += br.y + im * br.b_half
        Y[br.t, br.t] += br.y + im * br.b_half
        Y[br.f, br.t] -= br.y
        Y[br.t, br.f] -= br.y
    end
    return Y
end

"""
    NetworkData(bustype, pd, qd, pg, vm_setpoint, branches)

A small hand-coded network in per unit. `bustype` uses the MATPOWER convention
(1 = PQ, 2 = PV, 3 = slack); exactly one slack bus is required. `pd`/`qd` are base
loads, `pg` base generator setpoints, `vm_setpoint` the fixed voltage magnitudes at
PV and slack buses.
"""
struct NetworkData
    n::Int
    bustype::Vector{Int}
    pd::Vector{Float64}
    qd::Vector{Float64}
    pg::Vector{Float64}
    vm_setpoint::Vector{Float64}
    Y::Matrix{ComplexF64}
    branches::Vector{Branch}
end

function NetworkData(
    bustype::AbstractVector{Int},
    pd::AbstractVector{<:Real},
    qd::AbstractVector{<:Real},
    pg::AbstractVector{<:Real},
    vm_setpoint::AbstractVector{<:Real},
    branches::AbstractVector{Branch},
)
    n = length(bustype)
    all(t -> t in (1, 2, 3), bustype) ||
        throw(ArgumentError("bustype entries must be 1 (PQ), 2 (PV) or 3 (slack)"))
    count(==(3), bustype) == 1 || throw(ArgumentError("exactly one slack bus is required"))
    length(pd) == length(qd) == length(pg) == length(vm_setpoint) == n ||
        throw(DimensionMismatch("bus data vectors must all have length $(n)"))
    all(br -> 1 <= br.f <= n && 1 <= br.t <= n, branches) ||
        throw(ArgumentError("branch endpoints must be valid bus indices"))
    return NetworkData(
        n,
        collect(Int, bustype),
        collect(Float64, pd),
        collect(Float64, qd),
        collect(Float64, pg),
        collect(Float64, vm_setpoint),
        build_ybus(n, collect(Branch, branches)),
        collect(Branch, branches),
    )
end

"""
    ReferenceBackend(net::NetworkData; tol = 1e-8, maxiter = 20)

The reference [`AbstractBackend`](@ref): full-Newton AC power flow in polar form
with an analytic dense Jacobian. `tol` is on the infinity-norm of the power
mismatch.
"""
struct ReferenceBackend <: AbstractBackend
    net::NetworkData
    tol::Float64
    maxiter::Int
end

ReferenceBackend(net::NetworkData; tol::Real = 1e-8, maxiter::Integer = 20) =
    ReferenceBackend(net, tol, maxiter)

"""
    RefState

Mutable solver state of the [`ReferenceBackend`](@ref): the polar voltage solution,
working copies of the injections, and the component slots resolved once by
[`init_state`](@ref).
"""
mutable struct RefState
    vm::Vector{Float64}
    va::Vector{Float64}
    pd::Vector{Float64}
    qd::Vector{Float64}
    pg::Vector{Float64}
    slots::Vector{Tuple{Symbol,Int}}   # injection j writes (field, bus)
    ang_idx::Vector{Int}               # buses with a P equation / angle unknown
    pq_idx::Vector{Int}                # buses with a Q equation / magnitude unknown
end

# Demonstrates: refs are resolved to internal slots ONCE, and unresolvable refs
# error loudly (the backend's half of the spec validation rules).
function init_state(b::ReferenceBackend, refs::AbstractVector{ComponentRef})
    net = b.net
    slots = Vector{Tuple{Symbol,Int}}(undef, length(refs))
    for (j, ref) in enumerate(refs)
        bus = tryparse(Int, ref.id)
        (bus === nothing || !(1 <= bus <= net.n)) && throw(
            ArgumentError("component id $(repr(ref.id)) is not a bus number in 1:$(net.n)"),
        )
        field = if ref.kind == :load && ref.field in (:pd, :qd)
            ref.field
        elseif ref.kind == :gen && ref.field == :pg
            ref.field
        else
            throw(
                ArgumentError(
                    "unsupported component reference $(ref): the reference backend " *
                    "knows (:load, id, :pd|:qd) and (:gen, id, :pg)",
                ),
            )
        end
        if net.bustype[bus] == 3
            throw(
                ArgumentError(
                    "cannot assign an injection at slack bus $(bus): the slack " *
                    "injection is an outcome of the solve, the value would be ignored",
                ),
            )
        end
        if field == :qd && net.bustype[bus] == 2
            throw(
                ArgumentError(
                    "cannot assign reactive load at PV bus $(bus): its reactive " *
                    "injection is an outcome of the solve, the value would be ignored",
                ),
            )
        end
        slots[j] = (field, bus)
    end
    ang_idx = findall(t -> t != 3, net.bustype)
    pq_idx = findall(==(1), net.bustype)
    return RefState(
        copy(net.vm_setpoint),
        zeros(net.n),
        copy(net.pd),
        copy(net.qd),
        copy(net.pg),
        slots,
        ang_idx,
        pq_idx,
    )
end

# Demonstrates: the per-sample injection write is a plain slot write, ordered as the
# refs passed to init_state.
function set_injections!(state::RefState, ::ReferenceBackend, x::AbstractVector{<:Real})
    length(x) == length(state.slots) || throw(
        DimensionMismatch(
            "injection vector has length $(length(x)), expected $(length(state.slots))",
        ),
    )
    @inbounds for j in eachindex(state.slots)
        field, bus = state.slots[j]
        if field == :pd
            state.pd[bus] = x[j]
        elseif field == :qd
            state.qd[bus] = x[j]
        else
            state.pg[bus] = x[j]
        end
    end
    return state
end

# Complex bus voltages and injections S = V .* conj(Y * V) at the current iterate.
function bus_injections(state::RefState, net::NetworkData)
    V = state.vm .* cis.(state.va)
    return V .* conj.(net.Y * V), V
end

function mismatch!(f::Vector{Float64}, state::RefState, net::NetworkData)
    S, _ = bus_injections(state, net)
    na = length(state.ang_idx)
    @inbounds for (row, i) in enumerate(state.ang_idx)
        f[row] = (state.pg[i] - state.pd[i]) - real(S[i])
    end
    @inbounds for (row, i) in enumerate(state.pq_idx)
        f[na+row] = -state.qd[i] - imag(S[i])
    end
    return f
end

# Standard polar power flow Jacobian, dense. Row/column order matches the mismatch:
# [P equations at ang_idx; Q equations at pq_idx] × [va at ang_idx; vm at pq_idx].
function jacobian(state::RefState, net::NetworkData)
    S, _ = bus_injections(state, net)
    vm, va = state.vm, state.va
    G, B = real(net.Y), imag(net.Y)
    ang, pq = state.ang_idx, state.pq_idx
    na, nq = length(ang), length(pq)
    J = zeros(na + nq, na + nq)
    for (row, i) in enumerate(ang), (col, j) in enumerate(ang)
        if i == j
            J[row, col] = -imag(S[i]) - B[i, i] * vm[i]^2
        else
            th = va[i] - va[j]
            J[row, col] = vm[i] * vm[j] * (G[i, j] * sin(th) - B[i, j] * cos(th))
        end
    end
    for (row, i) in enumerate(ang), (col, j) in enumerate(pq)
        if i == j
            J[row, na+col] = real(S[i]) / vm[i] + G[i, i] * vm[i]
        else
            th = va[i] - va[j]
            J[row, na+col] = vm[i] * (G[i, j] * cos(th) + B[i, j] * sin(th))
        end
    end
    for (row, i) in enumerate(pq), (col, j) in enumerate(ang)
        if i == j
            J[na+row, col] = real(S[i]) - G[i, i] * vm[i]^2
        else
            th = va[i] - va[j]
            J[na+row, col] = -vm[i] * vm[j] * (G[i, j] * cos(th) + B[i, j] * sin(th))
        end
    end
    for (row, i) in enumerate(pq), (col, j) in enumerate(pq)
        if i == j
            J[na+row, na+col] = imag(S[i]) / vm[i] - B[i, i] * vm[i]
        else
            th = va[i] - va[j]
            J[na+row, na+col] = vm[i] * (G[i, j] * sin(th) - B[i, j] * cos(th))
        end
    end
    return J
end

# Demonstrates: `warmstart === nothing` resets to a deterministic flat start;
# a previously solved RefState warm-starts the iteration; divergence returns
# `SolveInfo(converged = false, ...)` and never throws, and the state remains
# usable for the next (cold) solve.
function solve!(state::RefState, b::ReferenceBackend; warmstart = nothing)
    net = b.net
    if warmstart === nothing
        @inbounds for i = 1:net.n
            state.va[i] = 0.0
            state.vm[i] = net.bustype[i] == 1 ? 1.0 : net.vm_setpoint[i]
        end
    else
        warmstart isa RefState ||
            throw(ArgumentError("warmstart must be a RefState of the same backend"))
        copyto!(state.vm, warmstart.vm)
        copyto!(state.va, warmstart.va)
    end
    na = length(state.ang_idx)
    f = Vector{Float64}(undef, na + length(state.pq_idx))
    residual = Inf
    for it = 0:b.maxiter
        mismatch!(f, state, net)
        residual = norm(f, Inf)
        isfinite(residual) || return SolveInfo(false, it, residual)
        residual < b.tol && return SolveInfo(true, it, residual)
        it == b.maxiter && break
        F = lu(jacobian(state, net); check = false)
        issuccess(F) || return SolveInfo(false, it, residual)
        step = F \ f
        @inbounds for (row, i) in enumerate(state.ang_idx)
            state.va[i] += step[row]
        end
        @inbounds for (row, i) in enumerate(state.pq_idx)
            state.vm[i] += step[na+row]
        end
    end
    return SolveInfo(false, b.maxiter, residual)
end

supports_warmstart(::ReferenceBackend) = true

extract(state::RefState, ::ReferenceBackend, q::VoltageMagnitude) = state.vm[q.bus]
extract(state::RefState, ::ReferenceBackend, q::VoltageAngle) = state.va[q.bus]

function extract(state::RefState, b::ReferenceBackend, q::BranchActivePower)
    for br in b.net.branches
        f, t = if (br.f, br.t) == (q.from, q.to)
            br.f, br.t
        elseif (br.t, br.f) == (q.from, q.to)
            br.t, br.f
        else
            continue
        end
        Vf = state.vm[f] * cis(state.va[f])
        Vt = state.vm[t] * cis(state.va[t])
        return real(Vf * conj((Vf - Vt) * br.y + Vf * im * br.b_half))
    end
    throw(ArgumentError("no branch between buses $(q.from) and $(q.to)"))
end
