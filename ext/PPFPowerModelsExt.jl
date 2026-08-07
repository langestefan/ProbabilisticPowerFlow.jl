module PPFPowerModelsExt

import PowerModels as PM
import ProbabilisticPowerFlow as PPF
import ProbabilisticPowerFlow:
    ComponentRef, SolveInfo, VoltageMagnitude, VoltageAngle, BranchActivePower

# PowerModels bus_type codes: 1 is PQ, 2 is PV, 3 is the reference bus, and 4 is
# inactive. The predicates take the raw code, plus a bus dictionary where the
# call sites have one.
is_pq_bus(t::Integer) = t == 1
is_pv_bus(t::Integer) = t == 2
is_slack_bus(t::Integer) = t == 3
is_pv_bus(bus::AbstractDict) = is_pv_bus(bus["bus_type"]::Int)
is_slack_bus(bus::AbstractDict) = is_slack_bus(bus["bus_type"]::Int)

"""
    PMBackend

Backend built by [`ProbabilisticPowerFlow.PowerModelsBackend`](@ref). It holds a
pristine deep copy of the validated PowerModels network data dictionary. Newton
iterations on a sparse admittance matrix, with the solve loop chosen by the
`solver` field: the bundled NLsolve path for `:nlsolve`, or a cached
NonlinearSolve.jl loop when a NonlinearSolve algorithm is passed and the
`PPFNonlinearSolveExt` extension is loaded.
"""
struct PMBackend{S} <: PPF.AbstractBackend
    data::Dict{String,Any}
    solver::S
    tol::Float64
    maxiter::Int
    branch_lookup::Dict{Tuple{Int,Int},Tuple{String,Bool}}
    ambiguous_pairs::Set{Tuple{Int,Int}}
end

PPF.supports_pf_solver(solver::Symbol) = solver === :nlsolve

function PPF.PowerModelsBackend(
    data::AbstractDict;
    solver = :nlsolve,
    tol::Real = 1e-8,
    maxiter::Integer = 100,
)
    PPF.supports_pf_solver(solver) || throw(
        ArgumentError(
            "unknown power flow solver $(repr(solver)). Use :nlsolve for the " *
            "bundled path, or pass a NonlinearSolve.jl algorithm after running " *
            "`using NonlinearSolve`.",
        ),
    )
    for table in ("bus", "load", "gen", "branch")
        haskey(data, table) || throw(
            ArgumentError(
                "expected a PowerModels network data dictionary, missing table " *
                "$(repr(table))",
            ),
        )
    end
    get(data, "per_unit", false) == true ||
        throw(ArgumentError("PowerModels data must be in per-unit"))
    get(data, "multinetwork", false) == false ||
        throw(ArgumentError("multinetwork data is not supported"))
    for table in ("dcline", "switch")
        isempty(get(data, table, Dict{String,Any}())) || throw(
            ArgumentError(
                "networks with a non-empty $(repr(table)) table are not supported " *
                "by the native PowerModels power flow",
            ),
        )
    end

    n_ref = count(is_slack_bus, values(data["bus"]))
    n_ref >= 1 || throw(ArgumentError("at least one reference bus is required"))
    gen_buses = Set(g["gen_bus"] for g in values(data["gen"]) if g["gen_status"] != 0)
    for bus in values(data["bus"])
        if (is_pv_bus(bus) || is_slack_bus(bus)) && !(bus["index"] in gen_buses)
            throw(
                ArgumentError(
                    "bus $(bus["index"]) has bus_type $(bus["bus_type"]) but no " *
                    "active generator",
                ),
            )
        end
    end

    branch_lookup = Dict{Tuple{Int,Int},Tuple{String,Bool}}()
    ambiguous_pairs = Set{Tuple{Int,Int}}()
    for (id, br) in data["branch"]
        br["br_status"] == 0 && continue
        f = br["f_bus"]::Int
        t = br["t_bus"]::Int
        for (key, at_from) in (((f, t), true), ((t, f), false))
            if haskey(branch_lookup, key)
                push!(ambiguous_pairs, key)
            else
                branch_lookup[key] = (String(id), at_from)
            end
        end
    end

    work = Dict{String,Any}(k => v for (k, v) in deepcopy(data))
    return PMBackend(
        work,
        solver,
        Float64(tol),
        Int(maxiter),
        branch_lookup,
        ambiguous_pairs,
    )
end

"""
    PMState

Mutable solver state of the [`PMBackend`](@ref): a working copy of the network data
dictionary, the injection slots resolved once by `init_state`, a lazy cache of
branch flows computed on the first `BranchActivePower` extract after a solve, and
the `PowerFlowData` built by PowerModels on the first solve and reused afterwards.
The admittance matrix, the Jacobian sparsity pattern, and the solver buffers in it
depend only on topology and setpoints, which the backend contract holds fixed, so
only the net bus injections are recomputed per solve.
"""
mutable struct PMState
    data::Dict{String,Any}
    slots::Vector{Tuple{Dict{String,Any},String}}
    solved::Bool
    flows::Union{Nothing,Dict{String,Any}}
    pf_data::Union{Nothing,PM.PowerFlowData}
    # snapshot of the net-injection and setpoint arrays at instantiate time plus
    # each slot's position in them, so the per-solve refresh is a copy and an
    # O(slots) update instead of a full Dict-based recomputation
    p_base::Vector{Float64}
    q_base::Vector{Float64}
    vm_base::Vector{Float64}
    slot_rows::Vector{Tuple{Int,Bool,Float64}}   # (row, is_p, sign)
    x_base::Vector{Float64}
    # used only by the NonlinearSolve solver path: the reusable nonlinear cache
    # and the last converged solution vector for warm starts
    solver_cache::Any
    last_x::Vector{Float64}
    has_last::Bool
end

function PPF.init_state(b::PMBackend, refs::AbstractVector{ComponentRef})
    work = deepcopy(b.data)
    slots = Vector{Tuple{Dict{String,Any},String}}(undef, length(refs))
    for (j, ref) in enumerate(refs)
        if ref.kind == :load && (ref.field == :pd || ref.field == :qd)
            table, buskey, statuskey = "load", "load_bus", "status"
        elseif ref.kind == :gen && ref.field == :pg
            table, buskey, statuskey = "gen", "gen_bus", "gen_status"
        else
            throw(
                ArgumentError(
                    "unsupported component reference $(ref): the PowerModels " *
                    "backend knows (:load, id, :pd|:qd) and (:gen, id, :pg)",
                ),
            )
        end
        comp = get(work[table], ref.id, nothing)
        comp === nothing && throw(
            ArgumentError(
                "component id $(repr(ref.id)) does not exist in the $(repr(table)) " *
                "table",
            ),
        )
        comp[statuskey] == 0 && throw(
            ArgumentError(
                "component id $(repr(ref.id)) in table $(repr(table)) is inactive. " *
                "An injection assigned to it would be silently ignored.",
            ),
        )
        bus = work["bus"]["$(comp[buskey])"]
        is_slack_bus(bus) && throw(
            ArgumentError(
                "cannot assign an injection at the slack bus $(comp[buskey]). The " *
                "slack balances the network, so the value would be silently ignored.",
            ),
        )
        if ref.field == :qd && is_pv_bus(bus)
            throw(
                ArgumentError(
                    "cannot assign reactive load at PV bus $(comp[buskey]). The " *
                    "voltage setpoint absorbs it, so the value would be silently " *
                    "ignored.",
                ),
            )
        end
        slots[j] = (comp, String(ref.field))
    end
    return PMState(
        work,
        slots,
        false,
        nothing,
        nothing,
        Float64[],
        Float64[],
        Float64[],
        Tuple{Int,Bool,Float64}[],
        Float64[],
        nothing,
        Float64[],
        false,
    )
end

function PPF.set_injections!(state::PMState, ::PMBackend, x::AbstractVector{<:Real})
    length(x) == length(state.slots) || throw(
        DimensionMismatch(
            "injection vector has length $(length(x)), expected $(length(state.slots))",
        ),
    )
    for j in eachindex(state.slots)
        comp, field = state.slots[j]
        comp[field] = Float64(x[j])
    end
    return state
end

# Replicates the solution mapping of PowerModels.compute_ac_pf: PQ buses read vm and
# va from the solution vector, PV buses read va only, and the slack keeps its
# setpoint magnitude with va fixed at 0.0 by the native solver.
function write_solution!(data::Dict{String,Any}, pf_data, x::AbstractVector{Float64})
    for (i, bid) in enumerate(pf_data.am.idx_to_bus)
        bus = data["bus"]["$(bid)"]
        t = pf_data.bus_type_idx[i]
        if is_pq_bus(t)
            bus["vm"] = x[2i-1]
            bus["va"] = x[2i]
        elseif is_pv_bus(t)
            bus["va"] = x[2i]
        elseif is_slack_bus(t)
            bus["va"] = 0.0
        end
    end
    return data
end

# Runs once, right after instantiate_pf_data built its arrays from the current
# network data: snapshots the net-injection and voltage-setpoint arrays and
# resolves every slot to its row in them. Setpoints and every non-slot injection
# are fixed by the backend contract, so the snapshot stays valid for the life of
# the state.
function bake_deltas!(state::PMState)
    pf = state.pf_data
    state.p_base = copy(pf.p_delta_base_idx)
    state.q_base = copy(pf.q_delta_base_idx)
    state.vm_base = copy(pf.vm_idx)
    resize!(state.slot_rows, length(state.slots))
    resize!(state.x_base, length(state.slots))
    for (j, (comp, field)) in enumerate(state.slots)
        bus = haskey(comp, "load_bus") ? comp["load_bus"] : comp["gen_bus"]
        row = pf.am.bus_to_idx[bus]
        # p_delta is pg minus pd and the idx arrays hold the sign-flipped delta,
        # so pd and qd enter with +1 and pg with -1
        is_p = field != "qd"
        sign = field == "pg" ? -1.0 : 1.0
        state.slot_rows[j] = (row, is_p, sign)
        state.x_base[j] = comp[field]::Float64
    end
    return state
end

# Restores the PowerFlowData to the state instantiate_pf_data built, then applies
# the slot differences. An O(buses) copy plus an O(slots) update, replacing the
# Dict-heavy full recomputation through calc_bus_injection, which allocated about
# 2 MB per solve on a 1354-bus case and throttled concurrent sampling. The
# working buffers are reset because the solver mutates them in place and its
# Jacobian callback reads vm_idx and va_idx directly, so leftovers from the
# previous solve would make the Newton path history dependent.
function refresh_pf_data!(state::PMState)
    pf = state.pf_data
    copyto!(pf.p_delta_base_idx, state.p_base)
    copyto!(pf.q_delta_base_idx, state.q_base)
    for (j, (comp, field)) in enumerate(state.slots)
        row, is_p, sign = state.slot_rows[j]
        diff = sign * (comp[field]::Float64 - state.x_base[j])
        if is_p
            pf.p_delta_base_idx[row] += diff
        else
            pf.q_delta_base_idx[row] += diff
        end
    end
    fill!(pf.x0, 0.0)
    fill!(pf.p_inject_idx, 0.0)
    fill!(pf.q_inject_idx, 0.0)
    copyto!(pf.vm_idx, state.vm_base)
    fill!(pf.va_idx, 0.0)
    return state
end

function PPF.solve!(state::PMState, b::PMBackend; warmstart = nothing)
    state.flows = nothing
    state.solved = false
    if warmstart !== nothing && !(warmstart isa PMState)
        throw(
            ArgumentError(
                "warmstart must be a previously solved state of the PowerModels " *
                "backend, got $(typeof(warmstart))",
            ),
        )
    end
    info = try
        if state.pf_data === nothing
            state.pf_data = PM.instantiate_pf_data(state.data)
            bake_deltas!(state)
        else
            refresh_pf_data!(state)
        end
        info, xsol = PPF.run_pf_solver!(state, b, b.solver, warmstart)
        info.converged &&
            xsol !== nothing &&
            write_solution!(state.data, state.pf_data, xsol)
        info
    catch err
        # Solvers throw on non-finite iterates and singular Jacobians. The
        # contract records divergence as data instead. A failed solve leaves no
        # stale state behind: the deltas are recomputed and the starting point is
        # fully rewritten on the next solve.
        err isa InterruptException && rethrow()
        SolveInfo(false, 0, NaN)
    end
    state.solved = info.converged
    return info
end

# The bundled solver path: PowerModels' _compute_ac_pf through NLsolve. Warm
# starts go through the vm_start and va_start keys, which the non-flat starting
# point reads.
function PPF.run_pf_solver!(state, b, ::Symbol, warmstart)
    flat = warmstart === nothing
    if !flat
        for (i, bus) in state.data["bus"]
            wbus = warmstart.data["bus"][i]
            bus["vm_start"] = wbus["vm"]
            bus["va_start"] = wbus["va"]
        end
    end
    res = PM._compute_ac_pf(
        state.pf_data;
        flat_start = flat,
        ftol = b.tol,
        iterations = b.maxiter,
    )
    converged = res.x_converged || res.f_converged
    return SolveInfo(converged, res.iterations, res.residual_norm), res.zero
end

PPF.supports_warmstart(::PMBackend) = true

function bus_entry(state::PMState, bus::Int)
    entry = get(state.data["bus"], string(bus), nothing)
    entry === nothing && throw(ArgumentError("no bus $(bus) in the network"))
    return entry
end

PPF.extract(s::PMState, ::PMBackend, q::VoltageMagnitude) =
    Float64(bus_entry(s, q.bus)["vm"])
PPF.extract(s::PMState, ::PMBackend, q::VoltageAngle) = Float64(bus_entry(s, q.bus)["va"])

function PPF.extract(s::PMState, b::PMBackend, q::BranchActivePower)
    key = (q.from, q.to)
    key in b.ambiguous_pairs && throw(
        ArgumentError(
            "parallel branches between buses $(q.from) and $(q.to) make " *
            "BranchActivePower ambiguous",
        ),
    )
    entry = get(b.branch_lookup, key, nothing)
    entry === nothing &&
        throw(ArgumentError("no branch between buses $(q.from) and $(q.to)"))
    id, at_from = entry
    if s.flows === nothing
        s.flows = PM.calc_branch_flow_ac(s.data)
    end
    flow = s.flows["branch"][id]
    return Float64(at_from ? flow["pf"] : flow["pt"])
end

# Display. The network data dictionary is megabytes, so neither the backend nor the
# state ever prints it. See src/show.jl for the tree helpers.
network_counts(data) = [
    "buses: $(length(data["bus"]))",
    "branches: $(length(data["branch"]))",
    "generators: $(length(data["gen"]))",
    "loads: $(length(data["load"]))",
]

Base.show(io::IO, b::PMBackend) =
    print(io, "PowerModelsBackend(", length(b.data["bus"]), " buses)")

Base.show(io::IO, ::MIME"text/plain", b::PMBackend) = PPF.show_tree(
    io,
    "PowerModelsBackend",
    [
        "network: $(get(b.data, "name", "unnamed"))" => network_counts(b.data),
        "solver: $(PPF.solver_name(b.solver))",
        "tol: $(b.tol)",
        "maxiter: $(b.maxiter)",
    ],
)

Base.show(io::IO, s::PMState) = print(
    io,
    "PMState(",
    length(s.data["bus"]),
    " buses, ",
    length(s.slots),
    " injection slots, ",
    s.solved ? "solved" : "unsolved",
    ")",
)

Base.show(io::IO, ::MIME"text/plain", s::PMState) = PPF.show_tree(
    io,
    "PowerModelsBackend state",
    [
        "network: $(get(s.data, "name", "unnamed"))" => network_counts(s.data),
        "injection slots: $(length(s.slots))",
        "solved: $(s.solved)",
        "solver data built: $(s.pf_data !== nothing)",
        "warm start available: $(s.has_last)",
    ],
)

end
