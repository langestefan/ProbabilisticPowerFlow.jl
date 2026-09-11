module PPFPowerModelsExt

import PowerModels as PM
import ProbabilisticPowerFlow as PPF
import ProbabilisticPowerFlow:
    ComponentRef,
    ComponentField,
    SolveInfo,
    VoltageMagnitude,
    VoltageAngle,
    BranchActivePower,
    BranchReactivePower,
    PowerModelsBackend

is_pq_bus(t::Integer) = t == 1
is_pv_bus(t::Integer) = t == 2
is_slack_bus(t::Integer) = t == 3
is_pv_bus(bus::AbstractDict) = is_pv_bus(bus["bus_type"]::Int)
is_slack_bus(bus::AbstractDict) = is_slack_bus(bus["bus_type"]::Int)

function PPF.PowerModelsBackend(data::AbstractDict; alg = PM.NativeNewton())
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

    # More informative error than the error from calc_bus_injection
    for table in ("dcline", "switch")
        isempty(get(data, table, Dict{String, Any}())) || throw(
            ArgumentError(
                "networks with a non-empty $(repr(table)) table are not supported by " *
                    "the native PowerModels power flow",
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
                    "bus $(bus["index"]) has bus_type $(bus["bus_type"]) but no active " *
                        "generator",
                ),
            )
        end
    end

    # Look up branch quantitys of interest by bus pair, and record ambiguous pairs
    branch_lookup = Dict{Tuple{Int, Int}, Tuple{String, Bool}}()
    ambiguous_pairs = Set{Tuple{Int, Int}}()
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

    work = Dict{String, Any}(k => v for (k, v) in deepcopy(data))
    return PowerModelsBackend(work, alg, branch_lookup, ambiguous_pairs)
end

# direct dispatch on case string
PPF.PowerModelsBackend(filename::AbstractString; alg = PM.NativeNewton()) =
    PPF.PowerModelsBackend(PM.parse_file(filename), alg = alg)

"""
    PMState

Mutable solver state of a [`PowerModelsBackend`](@ref).
"""
mutable struct PMState{S, D}
    data::Dict{String, Any}
    sys::S
    pf_data::D
    net_base::Vector{Float64}
    cold_start::Vector{Float64}
    slot_rows::Vector{Tuple{Int, Bool, Float64}}
    solved::Bool
    flows::Union{Nothing, Dict{String, Any}}
    last_solution::Vector{Float64}
    has_solution::Bool
end

function slot_sign(field::ComponentField.T)
    is_p = field !== ComponentField.Qd
    sign = field === ComponentField.Pg ? -1.0 : 1.0
    return is_p, sign
end

function resolve_slot(work::Dict{String, Any}, ref::ComponentRef)
    if ref.field === ComponentField.Pd || ref.field === ComponentField.Qd
        table, buskey, statuskey = "load", "load_bus", "status"
    elseif ref.field === ComponentField.Pg
        table, buskey, statuskey = "gen", "gen_bus", "gen_status"
    else
        throw(
            ArgumentError(
                "unsupported component reference $(ref). The PowerModels backend can " *
                    "only assign quantities carried by the solver's parameter vector, " *
                    "which are Pd and Qd on loads and Pg on generators. A voltage setpoint " *
                    "is compiled into the residual function and reactive generation is " *
                    "solved for, so neither can be written per sample.",
            ),
        )
    end

    comp = get(work[table], string(ref.id), nothing)
    comp === nothing && throw(
        ArgumentError("component id $(ref.id) does not exist in the $(repr(table)) table"),
    )
    comp[statuskey] == 0 && throw(
        ArgumentError(
            "component id $(ref.id) in table $(repr(table)) is inactive. An injection " *
                "assigned to it would be silently ignored.",
        ),
    )

    busid = comp[buskey]::Int
    bus = work["bus"]["$(busid)"]
    is_slack_bus(bus) && throw(
        ArgumentError(
            "cannot assign an injection at the slack bus $(busid). The slack balances " *
                "the network, so the value would be silently ignored.",
        ),
    )
    if ref.field === ComponentField.Qd && is_pv_bus(bus)
        throw(
            ArgumentError(
                "cannot assign reactive load at PV bus $(busid). The voltage setpoint " *
                    "absorbs it, so the value would be silently ignored.",
            ),
        )
    end
    return busid, Float64(comp[field_key(ref.field)])
end

# the data dictionary key each assignable field is stored under
field_key(f::ComponentField.T) =
    f === ComponentField.Pd ? "pd" : f === ComponentField.Qd ? "qd" : "pg"

# Maps each assignment to its slot in the solver's parameter vector: the internal bus
# index, whether it is active or reactive, and its sign. Also returns each assigned
# component's original value, which fixed_injections needs.
function map_slots(work, pf_data, refs)
    slot_rows = Vector{Tuple{Int, Bool, Float64}}(undef, length(refs))
    originals = Vector{Float64}(undef, length(refs))
    for (j, ref) in enumerate(refs)
        busid, originals[j] = resolve_slot(work, ref)
        is_p, sign = slot_sign(ref.field)
        slot_rows[j] = (pf_data.am.bus_to_idx[busid], is_p, sign)
    end
    return slot_rows, originals
end

# PowerModels builds sys.p0 from the network file, so every bus total already includes the
# original value of each assigned component. set_injections! adds the sampled values on
# top, so the originals are subtracted here once to avoid counting them twice. What is
# left is the fixed injection at each bus.
function fixed_injections(p0, slot_rows, originals)
    net_base = copy(p0)
    for ((row, is_p, sign), original) in zip(slot_rows, originals)
        if is_p
            net_base[2row - 1] -= sign * original
        else
            net_base[2row] -= sign * original
        end
    end
    return net_base
end

function PPF.init_state(b::PowerModelsBackend, refs::AbstractVector{ComponentRef})
    work = deepcopy(b.data)

    # write injections into the parameter vector
    pf_data = PM.instantiate_pf_data(work)
    sys = PM.build_pf_system(pf_data)

    slot_rows, originals = map_slots(work, pf_data, refs)
    net_base = fixed_injections(sys.p0, slot_rows, originals)

    return PMState(
        work,
        sys,
        pf_data,
        net_base,
        copy(sys.x0),   # cold_start
        slot_rows,
        false,
        nothing,
        copy(sys.x0),   # last_solution, only read once has_solution is true
        false,
    )
end

# set_injections! converts sampled injections into the solver's parameter vector sys.p0.
# It is called once per sample, before the solver is invoked.
#
# - `x` is the sample's injections, in the order of the refs passed to init_state
# - `net` is the solver's parameter vector `sys.p0` with two entries per bus: net active
#   injection at `2i-1` and net reactive at `2i`
# - `state.net_base` is the fixed (static) injection at each bus
# - `state.slot_rows` gives the mapping between the sample's injection vector and the
#   solver's parameter vector. Each entry is a tuple of the bus row index, whether the
#   injection is active or reactive, and the sign to apply: +1 for a load, -1 for a
#   generator
function PPF.set_injections!(
        state::PMState,
        ::PowerModelsBackend,
        x::AbstractVector{<:Real},
    )
    length(x) == length(state.slot_rows) || throw(
        DimensionMismatch(
            "injection vector has length $(length(x)), expected " *
                "$(length(state.slot_rows))",
        ),
    )

    # First we reset to the base network (static) injections
    net = state.sys.p0
    copyto!(net, state.net_base)

    # Now add the sample's injections on top
    @inbounds for j in eachindex(state.slot_rows)
        row, is_p, sign = state.slot_rows[j]
        if is_p
            net[2row - 1] += sign * x[j] # active power injection
        else
            net[2row] += sign * x[j] # reactive power injection
        end
    end
    return state
end

# PQ buses read vm and va from the solution vector
# PV buses read va only and keep their setpoint magnitude
# the slack keeps both its setpoint and a zero angle
function write_solution!(state::PMState, x::AbstractVector{Float64})
    pf = state.pf_data
    data = state.data
    for (i, bid) in enumerate(pf.am.idx_to_bus)
        bus = data["bus"]["$(bid)"]
        t = pf.bus_type_idx[i]
        if is_pq_bus(t)
            bus["vm"] = x[2i - 1]
            bus["va"] = x[2i]
        elseif is_pv_bus(t)
            bus["va"] = x[2i]
        elseif is_slack_bus(t)
            bus["va"] = 0.0
        end
    end
    return data
end

function PPF.solve!(state::PMState, b::PowerModelsBackend; warmstart = nothing)
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
        # sys.x0 is the solver's starting point, not an injection
        if warmstart === nothing || !warmstart.has_solution
            copyto!(state.sys.x0, state.cold_start)
        else
            copyto!(state.sys.x0, warmstart.last_solution)
        end

        sol = PM._solve_nl(state.sys, b.alg)
        if sol.converged
            # sol.x is the converged solver state, voltages and PV and slack unknowns
            copyto!(state.last_solution, sol.x)
            state.has_solution = true
            write_solution!(state, sol.x)
        end
        SolveInfo(sol.converged, sol.iterations, sol.residual_norm)
    catch err
        # Divergences are recorded
        err isa InterruptException && rethrow()
        SolveInfo(false, -1, Inf)
    end

    state.solved = info.converged
    return info
end

PPF.supports_warmstart(::PowerModelsBackend) = true

function bus_entry(state::PMState, bus::Int)
    entry = get(state.data["bus"], string(bus), nothing)
    entry === nothing && throw(ArgumentError("no bus $(bus) in the network"))
    return entry
end

PPF.extract(s::PMState, ::PowerModelsBackend, q::VoltageMagnitude) =
    Float64(bus_entry(s, q.bus)["vm"])

PPF.extract(s::PMState, ::PowerModelsBackend, q::VoltageAngle) =
    Float64(bus_entry(s, q.bus)["va"])

function branch_flow(s::PMState, b::PowerModelsBackend, from::Int, to::Int)
    key = (from, to)
    key in b.ambiguous_pairs && throw(
        ArgumentError(
            "parallel branches between buses $(from) and $(to) make a branch flow " *
                "quantity ambiguous",
        ),
    )
    entry = get(b.branch_lookup, key, nothing)
    entry === nothing && throw(ArgumentError("no branch between buses $(from) and $(to)"))

    id, at_from = entry
    if s.flows === nothing
        s.flows = PM.calc_branch_flow_ac(s.data)
    end
    return s.flows["branch"][id], at_from
end

function PPF.extract(s::PMState, b::PowerModelsBackend, q::BranchActivePower)
    flow, at_from = branch_flow(s, b, q.from, q.to)
    return Float64(at_from ? flow["pf"] : flow["pt"])
end

function PPF.extract(s::PMState, b::PowerModelsBackend, q::BranchReactivePower)
    flow, at_from = branch_flow(s, b, q.from, q.to)
    return Float64(at_from ? flow["qf"] : flow["qt"])
end

end
