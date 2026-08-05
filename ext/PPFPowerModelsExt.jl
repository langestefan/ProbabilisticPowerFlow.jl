module PPFPowerModelsExt

import PowerModels as PM
import ProbabilisticPowerFlow as PPF
import ProbabilisticPowerFlow:
    ComponentRef, SolveInfo, VoltageMagnitude, VoltageAngle, BranchActivePower

"""
    PMBackend

Backend built by [`ProbabilisticPowerFlow.PowerModelsBackend`](@ref). It holds a
pristine deep copy of the validated PowerModels network data dictionary. Solves run
PowerModels' native AC power flow: Newton iterations on a sparse admittance matrix.
"""
struct PMBackend <: PPF.AbstractBackend
    data::Dict{String,Any}
    tol::Float64
    maxiter::Int
    branch_lookup::Dict{Tuple{Int,Int},Tuple{String,Bool}}
    ambiguous_pairs::Set{Tuple{Int,Int}}
end

function PPF.PowerModelsBackend(
    data::AbstractDict;
    tol::Real = 1e-8,
    maxiter::Integer = 100,
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

    n_ref = count(b -> b["bus_type"] == 3, values(data["bus"]))
    n_ref == 1 ||
        throw(ArgumentError("exactly one reference bus is required, got $(n_ref)"))
    gen_buses = Set(g["gen_bus"] for g in values(data["gen"]) if g["gen_status"] != 0)
    for bus in values(data["bus"])
        t = bus["bus_type"]
        if (t == 2 || t == 3) && !(bus["index"] in gen_buses)
            throw(
                ArgumentError(
                    "bus $(bus["index"]) has bus_type $(t) but no active generator",
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
    return PMBackend(work, Float64(tol), Int(maxiter), branch_lookup, ambiguous_pairs)
end

"""
    PMState

Mutable solver state of the [`PMBackend`](@ref): a working copy of the network data
dictionary, the injection slots resolved once by `init_state`, and a lazy cache of
branch flows computed on the first `BranchActivePower` extract after a solve.
"""
mutable struct PMState
    data::Dict{String,Any}
    slots::Vector{Tuple{Dict{String,Any},String}}
    solved::Bool
    flows::Union{Nothing,Dict{String,Any}}
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
        bus["bus_type"] == 3 && throw(
            ArgumentError(
                "cannot assign an injection at the slack bus $(comp[buskey]). The " *
                "slack balances the network, so the value would be silently ignored.",
            ),
        )
        if ref.field == :qd && bus["bus_type"] == 2
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
    return PMState(work, slots, false, nothing)
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
        if t == 1
            bus["vm"] = x[2i-1]
            bus["va"] = x[2i]
        elseif t == 2
            bus["va"] = x[2i]
        elseif t == 3
            bus["va"] = 0.0
        end
    end
    return data
end

function PPF.solve!(state::PMState, b::PMBackend; warmstart = nothing)
    state.flows = nothing
    state.solved = false
    flat = warmstart === nothing
    if !flat
        warmstart isa PMState || throw(
            ArgumentError(
                "warmstart must be a previously solved state of the PowerModels " *
                "backend, got $(typeof(warmstart))",
            ),
        )
        for (i, bus) in state.data["bus"]
            wbus = warmstart.data["bus"][i]
            bus["vm_start"] = wbus["vm"]
            bus["va_start"] = wbus["va"]
        end
    end
    info = try
        pf_data = PM.instantiate_pf_data(state.data)
        res = PM._compute_ac_pf(
            pf_data;
            flat_start = flat,
            ftol = b.tol,
            iterations = b.maxiter,
        )
        converged = res.x_converged || res.f_converged
        converged && write_solution!(state.data, pf_data, res.zero)
        SolveInfo(converged, res.iterations, res.residual_norm)
    catch err
        # NLsolve throws on non-finite iterates and singular Jacobians. The
        # contract records divergence as data instead.
        err isa InterruptException && rethrow()
        SolveInfo(false, 0, NaN)
    end
    state.solved = info.converged
    return info
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

end
