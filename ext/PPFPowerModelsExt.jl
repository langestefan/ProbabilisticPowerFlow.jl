module PPFPowerModelsExt

import PowerModels as PM
import ProbabilisticPowerFlow as PPF
import ProbabilisticPowerFlow:
    ComponentField,
    ComponentRef,
    SolveInfo,
    VoltageMagnitude,
    VoltageAngle,
    BranchActivePower,
    BranchReactivePower

# PowerModels bus_type codes: 1 is PQ, 2 is PV, 3 is the reference bus, 4 inactive.
is_pv_bus(t::Integer) = t == 2
is_slack_bus(t::Integer) = t == 3
is_pv_bus(bus::AbstractDict) = is_pv_bus(bus["bus_type"]::Int)
is_slack_bus(bus::AbstractDict) = is_slack_bus(bus["bus_type"]::Int)

"""
    PMBackend

Backend built by [`ProbabilisticPowerFlow.PowerModelsBackend`](@ref). Holds a
pristine deep copy of the validated network data dictionary plus the branch lookup
used by the flow QoIs. It is never mutated after construction, so all its states are
independent and concurrent sampling needs nothing else.
"""
struct PMBackend{S} <: PPF.AbstractPFBackend
    data::Dict{String,Any}
    solver::S
    tol::Float64
    maxiter::Int
    branch_lookup::Dict{Tuple{Int,Int},Tuple{String,Bool}}
    ambiguous_pairs::Set{Tuple{Int,Int}}
end

function PPF.PowerModelsBackend(
    data::AbstractDict;
    solver = PM.NativeNewton(),
    tol::Real = 1e-8,
    maxiter::Integer = 50,
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
                "networks with a non-empty $(repr(table)) table are not supported by " *
                "the native PowerModels power flow",
            ),
        )
    end

    count(is_slack_bus, values(data["bus"])) >= 1 ||
        throw(ArgumentError("at least one reference bus is required"))
    # PowerModels' power flow needs generators and voltage-controlled buses to agree.
    # Both directions are checked here rather than left to instantiate_pf_data, whose
    # failure on the second one is a bare `@assert false`.
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
    for (id, gen) in data["gen"]
        gen["gen_status"] == 0 && continue
        bus = data["bus"]["$(gen["gen_bus"])"]
        (is_pv_bus(bus) || is_slack_bus(bus)) || throw(
            ArgumentError(
                "generator $(id) is at bus $(gen["gen_bus"]), which has bus_type " *
                "$(bus["bus_type"]). The PowerModels power flow represents generators " *
                "only at PV and reference buses. Model an injection at a PQ bus as a " *
                "negative load instead: assign Pd through an AffineTransform with a " *
                "negative scale.",
            ),
        )
    end

    # a branch QoI names its two buses, so the pair has to resolve to one branch.
    # Parallel branches make that ambiguous, and are recorded rather than guessed at.
    branch_lookup = Dict{Tuple{Int,Int},Tuple{String,Bool}}()
    ambiguous_pairs = Set{Tuple{Int,Int}}()
    for (id, br) in data["branch"]
        br["br_status"] == 0 && continue
        f, t = br["f_bus"]::Int, br["t_bus"]::Int
        for (key, at_from) in (((f, t), true), ((t, f), false))
            if haskey(branch_lookup, key)
                push!(ambiguous_pairs, key)
            else
                branch_lookup[key] = (String(id), at_from)
            end
        end
    end

    return PMBackend(
        Dict{String,Any}(deepcopy(data)),
        with_limits(solver, Float64(tol), Int(maxiter)),
        Float64(tol),
        Int(maxiter),
        branch_lookup,
        ambiguous_pairs,
    )
end

# PowerModels takes the iteration limit and tolerance on the algorithm object for its
# own NativeNewton, and as solve keyword arguments for everything else. The backend
# presents one contract, so the difference is absorbed here: NativeNewton is rebuilt
# once at construction, and every other algorithm gets the keywords per solve.
with_limits(alg::PM.NativeNewton, tol::Float64, maxiter::Int) =
    PM.NativeNewton(; maxiters = maxiter, abstol = tol, alg.linesearch, alg.maxstep)
with_limits(alg, ::Float64, ::Int) = alg

run_solver(sys, b::PMBackend{PM.NativeNewton}) = PM._solve_nl(sys, b.solver)
run_solver(sys, b::PMBackend) =
    PM._solve_nl(sys, b.solver; abstol = b.tol, maxiters = b.maxiter)

"""
    PMState

Mutable solver state of the [`PMBackend`](@ref). Owns a private copy of the network
data and its own `PowerFlowSystem`, whose residual and Jacobian closures capture that
copy's arrays. That ownership is what makes states independent.

`slots[j]` resolves assignment `j` to `(row in the parameter vector, sign, base
value)`, so `set_injections!` is arithmetic on a `Vector{Float64}` with no dictionary
work per sample.
"""
mutable struct PMState{S}
    data::Dict{String,Any}
    pf_data::PM.PowerFlowData
    sys::S
    slots::Vector{Tuple{Int,Float64,Float64}}
    p_base::Vector{Float64}
    x_cold::Vector{Float64}
    x_last::Vector{Float64}
    solved::Bool
    flows::Union{Nothing,Dict{String,Any}}
end

# The table, bus key and status key each supported field lives under, and the sign it
# enters the parameter vector with. p_delta holds the negated net injection, so a load
# raises it and a generator lowers it.
function field_info(f::ComponentField.T)
    f === ComponentField.Pd && return ("load", "load_bus", "status", "pd", true, 1.0)
    f === ComponentField.Qd && return ("load", "load_bus", "status", "qd", false, 1.0)
    f === ComponentField.Pg && return ("gen", "gen_bus", "gen_status", "pg", true, -1.0)
    throw(
        ArgumentError(
            "the PowerModels backend cannot assign $(f). It writes nodal injections, " *
            "so it knows Pd, Qd and Pg. Qg is an outcome of the solve at PV and slack " *
            "buses, and Vg and Vm are setpoints rather than injections.",
        ),
    )
end

function PPF.init_state(b::PMBackend, refs::AbstractVector{ComponentRef})
    work = Dict{String,Any}(deepcopy(b.data))
    pf_data = PM.instantiate_pf_data(work)
    sys = PM.build_pf_system(pf_data)
    bus_to_idx = pf_data.am.bus_to_idx

    slots = Vector{Tuple{Int,Float64,Float64}}(undef, length(refs))
    for (j, ref) in enumerate(refs)
        table, buskey, statuskey, key, is_p, sign = field_info(ref.field)

        comp = get(work[table], string(ref.id), nothing)
        comp === nothing &&
            throw(ArgumentError("no component $(ref.id) in the $(repr(table)) table"))
        comp[statuskey] == 0 && throw(
            ArgumentError(
                "component $(ref.id) in table $(repr(table)) is inactive, so an " *
                "injection assigned to it would be silently ignored",
            ),
        )

        bus_id = comp[buskey]::Int
        bus = work["bus"]["$(bus_id)"]
        is_slack_bus(bus) && throw(
            ArgumentError(
                "cannot assign $(ref.field) at slack bus $(bus_id). The slack balances " *
                "the network, so its injection is an outcome of the solve and the " *
                "value would be silently ignored.",
            ),
        )
        if ref.field === ComponentField.Qd && is_pv_bus(bus)
            throw(
                ArgumentError(
                    "cannot assign Qd at PV bus $(bus_id). The voltage setpoint absorbs " *
                    "reactive power, so the value would be silently ignored.",
                ),
            )
        end
        row = bus_to_idx[bus_id]
        slots[j] = (is_p ? 2row - 1 : 2row, sign, comp[key]::Float64)
    end

    return PMState(
        work,
        pf_data,
        sys,
        slots,
        copy(sys.p0),
        copy(sys.x0),
        copy(sys.x0),
        false,
        nothing,
    )
end

# p_base already holds each component's base contribution, so a slot writes only its
# difference from that base. The reset-then-accumulate order is what lets several
# assignments on one bus compose, which is how a constant-power-factor load works.
function PPF.set_injections!(state::PMState, ::PMBackend, x::AbstractVector{<:Real})
    length(x) == length(state.slots) || throw(
        DimensionMismatch(
            "injection vector has length $(length(x)), expected $(length(state.slots))",
        ),
    )
    p = state.sys.p0
    copyto!(p, state.p_base)
    @inbounds for j in eachindex(state.slots)
        row, sign, base = state.slots[j]
        p[row] += sign * (x[j] - base)
    end
    return state
end

function PPF.solve!(state::PMState, b::PMBackend; warmstart = nothing)
    state.flows = nothing
    state.solved = false

    if warmstart === nothing
        copyto!(state.sys.x0, state.x_cold)
    elseif warmstart isa PMState
        copyto!(state.sys.x0, warmstart.x_last)
    else
        throw(
            ArgumentError(
                "warmstart must be a previously solved state of the PowerModels " *
                "backend, got $(typeof(warmstart))",
            ),
        )
    end

    info = try
        sol = run_solver(state.sys, b)
        if sol.converged
            copyto!(state.x_last, sol.x)
            # the last residual evaluation is not necessarily at the solution, so the
            # bus state vectors are written from the returned point explicitly
            PM._update_pf_state!(state.pf_data, sol.x)
        end
        SolveInfo(sol.converged, sol.iterations, sol.residual_norm)
    catch err
        # solvers throw on non-finite iterates and singular Jacobians. The contract
        # records divergence as data instead. A failed solve leaves nothing stale
        # behind: p and x0 are fully rewritten before the next solve.
        err isa InterruptException && rethrow()
        SolveInfo(false, -1, Inf)
    end

    state.solved = info.converged
    return info
end

PPF.supports_warmstart(::PMBackend) = true

# After a diverged solve the bus state vectors hold the final Newton iterate, which
# is not a solution. Reading one would hand back a plausible-looking number, and a
# ViolationEvent would silently count it, so extraction refuses instead. The sample
# loop only extracts from converged solves, so this guards use at the prompt.
function check_solved(s::PMState)
    s.solved || throw(
        ArgumentError(
            "the last solve on this state did not converge, so there is no solution " *
            "to extract. Check SolveInfo.converged before extracting.",
        ),
    )
    return nothing
end

function bus_index(state::PMState, bus::Int)
    check_solved(state)
    idx = get(state.pf_data.am.bus_to_idx, bus, nothing)
    idx === nothing && throw(ArgumentError("no bus $(bus) in the network"))
    return idx
end

PPF.extract(s::PMState, ::PMBackend, q::VoltageMagnitude) =
    s.pf_data.vm_idx[bus_index(s, q.bus)]
PPF.extract(s::PMState, ::PMBackend, q::VoltageAngle) =
    s.pf_data.va_idx[bus_index(s, q.bus)]

# Branch flows are the one dictionary-heavy path left, so they are computed lazily and
# cached: a run that asks only for voltages never pays for them.
function branch_flows(s::PMState)
    check_solved(s)
    s.flows === nothing || return s.flows
    for (i, bid) in enumerate(s.pf_data.am.idx_to_bus)
        bus = s.data["bus"]["$(bid)"]
        bus["vm"] = s.pf_data.vm_idx[i]
        bus["va"] = s.pf_data.va_idx[i]
    end
    s.flows = PM.calc_branch_flow_ac(s.data)
    return s.flows
end

function branch_entry(s::PMState, b::PMBackend, from::Int, to::Int)
    key = (from, to)
    key in b.ambiguous_pairs && throw(
        ArgumentError(
            "parallel branches between buses $(from) and $(to) make the flow QoI " *
            "ambiguous",
        ),
    )
    entry = get(b.branch_lookup, key, nothing)
    entry === nothing && throw(ArgumentError("no branch between buses $(from) and $(to)"))
    id, at_from = entry
    return branch_flows(s)["branch"][id], at_from
end

function PPF.extract(s::PMState, b::PMBackend, q::BranchActivePower)
    flow, at_from = branch_entry(s, b, q.from, q.to)
    return Float64(at_from ? flow["pf"] : flow["pt"])
end

function PPF.extract(s::PMState, b::PMBackend, q::BranchReactivePower)
    flow, at_from = branch_entry(s, b, q.from, q.to)
    return Float64(at_from ? flow["qf"] : flow["qt"])
end

# Display. The network data dictionary is megabytes, so neither the backend nor the
# state ever prints it.
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
        "solver: $(nameof(typeof(b.solver)))",
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
    ],
)

end
