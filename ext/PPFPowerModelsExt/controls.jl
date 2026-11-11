# Controls inside the Newton solve.
#
# Every controlled bus is a PQ bus whose solver slot 2i-1 holds one unknown s. A slot map
# turns s into the bus voltage magnitude and a reactive injection, both with their
# derivatives in s. A generator under ReactiveLimits has its PV bus turned into a PQ bus
# for this. The system keeps its size and sparsity pattern, because J[2i, 2i-1] exists for
# every PQ bus and PowerModels stores full 2x2 blocks for every pair of neighbours.
#
# Under Complementarity, s is the control variable of the normal map and the voltage
# follows from it. Under Sigmoid, and for a snapped shunt, s is the voltage itself.
#
# PowerModels writes the reactive balance with generation entering negatively, so each
# injection is subtracted from F[2i].

# One generator's share of a limited bus
struct GenShare
    id::Int
    qmin::Float64
    qmax::Float64
end

# The summed reactive limits of every generator at one controlled bus
struct LimitTerm
    row::Int
    vset::Float64
    qmin::Float64
    qmax::Float64
    # the summed qg from the network data, clamped to the limits
    qstart::Float64
    # the voltage change per unit of reactive power beyond the limits
    scale::Float64
    gens::Vector{GenShare}
end

# A switched shunt, continuous until it is snapped to one of its steps
mutable struct ShuntTerm
    const id::Int
    const row::Int
    const vset::Float64
    const steps::Vector{Float64}
    # the voltage change per unit of susceptance beyond the steps
    const scale::Float64
    snapped::Bool
    b::Float64
end

# The per-state control data. The steepness changes during a Sigmoid homotopy.
mutable struct ControlState{M <: AbstractControlModel}
    const model::M
    const limits::Vector{LimitTerm}
    const shunts::Vector{ShuntTerm}
    steepness::Float64
end

function ControlState(model::AbstractControlModel, limits, shunts)
    if model isa Sigmoid
        steepness = model.steepness
    else
        steepness = 0.0
    end
    return ControlState(model, limits, shunts, steepness)
end

has_controls(c::ControlState) = !isempty(c.limits) || !isempty(c.shunts)

slot(t) = 2 * t.row - 1

function foreach_term(f, c::ControlState)
    foreach(f, c.limits)
    foreach(f, c.shunts)
    return nothing
end

# The falling logistic 1 / (1 + exp(z)) and its derivative in z. Both branches avoid
# overflow for a large steepness.
function falling_sigmoid(z::Float64)
    if z > 0
        e = exp(-z)
        s = e / (1 + e)
    else
        s = 1 / (1 + exp(z))
    end
    return s, -s * (1 - s)
end

# The value between lo and hi at voltage v, and its derivative in v
function smooth_level(lo, hi, vset, steepness, v)
    s, ds = falling_sigmoid(steepness * (v - vset))
    return lo + (hi - lo) * s, (hi - lo) * steepness * ds
end

# The normal map of the box [lo, hi]. Returns the value, its derivative, the voltage and
# its derivative, all in s. At a kink it takes the inside, which is the PV bus.
function box_map(lo, hi, vset, scale, s)
    if s > hi
        return hi, 0.0, vset - scale * (s - hi), -scale
    elseif s < lo
        return lo, 0.0, vset + scale * (lo - s), -scale
    end
    return s, 1.0, vset, 0.0
end

# The slot value on the normal map that gives voltage v, with the value nearest b
function box_slot(lo, hi, vset, scale, v, b)
    if v < vset
        return hi + (vset - v) / scale
    elseif v > vset
        return lo - (v - vset) / scale
    end
    return b
end

# The reactive output of a limited bus, the voltage, and their derivatives in s
function slot_map(t::LimitTerm, c::ControlState{Complementarity}, s)
    q, dq, v, dv = box_map(t.qmin, t.qmax, t.vset, t.scale, s)
    return v, dv, q, dq
end

function slot_map(t::LimitTerm, c::ControlState{Sigmoid}, s)
    q, dq = smooth_level(t.qmin, t.qmax, t.vset, c.steepness, s)
    return s, 1.0, q, dq
end

# The susceptance of a shunt, the voltage, and their derivatives in s
function shunt_level(t::ShuntTerm, c::ControlState{Complementarity}, s)
    if t.snapped
        return t.b, 0.0, s, 1.0
    end
    return box_map(first(t.steps), last(t.steps), t.vset, t.scale, s)
end

function shunt_level(t::ShuntTerm, c::ControlState{Sigmoid}, s)
    if t.snapped
        return t.b, 0.0, s, 1.0
    end
    b, db = smooth_level(first(t.steps), last(t.steps), t.vset, c.steepness, s)
    return b, db, s, 1.0
end

# A shunt injects Q = b v^2
function slot_map(t::ShuntTerm, c::ControlState, s)
    b, db, v, dv = shunt_level(t, c, s)
    return v, dv, b * v^2, db * v^2 + 2 * b * v * dv
end

# Copies the solver vector x into xv with each controlled slot replaced by its voltage
function slot_voltages!(xv, x, c::ControlState)
    copyto!(xv, x)
    foreach_term(c) do t
        v, _ = slot_map(t, c, x[slot(t)])
        xv[slot(t)] = v
    end
    return xv
end

function add_control_residual!(F, x, c::ControlState)
    foreach_term(c) do t
        _, _, q, _ = slot_map(t, c, x[slot(t)])
        F[2 * t.row] -= q
    end
    return F
end

# PowerModels filled column 2i-1 with the derivatives in the voltage, so the chain rule
# scales it by dv/ds before the injection's own derivative is added
function add_control_jacobian!(J, x, c::ControlState)
    foreach_term(c) do t
        col = slot(t)
        _, dv, _, dq = slot_map(t, c, x[col])
        for k in J.colptr[col]:(J.colptr[col + 1] - 1)
            J.nzval[k] *= dv
        end
        J[2 * t.row, col] -= dq
    end
    return J
end

# Wraps the PowerModels residual and Jacobian. The wrapped system shares x0 and p0 with
# the original, so set_injections! and warm starts work unchanged.
function controlled_system(sys::PM.PowerFlowSystem, c::ControlState)
    xv = similar(sys.x0)
    f! = function (F, x, p)
        sys.f!(F, slot_voltages!(xv, x, c), p)
        add_control_residual!(F, x, c)
        return nothing
    end
    j! = function (J, x, p)
        sys.j!(J, slot_voltages!(xv, x, c), p)
        add_control_jacobian!(J, x, c)
        return nothing
    end
    return PM.PowerFlowSystem(f!, j!, sys.x0, sys.p0, sys.jac_prototype)
end

function unsnap!(c::ControlState)
    for t in c.shunts
        t.snapped = false
    end
    return c
end

# Fixes every shunt at the step nearest to its continuous susceptance at solution x. A
# snapped slot holds the voltage, so x is rewritten to match.
function snap!(c::ControlState, x)
    for t in c.shunts
        b, _, v, _ = shunt_level(t, c, x[slot(t)])
        t.b = t.steps[argmin(abs.(t.steps .- b))]
        t.snapped = true
        x[slot(t)] = v
    end
    return x
end

# Rewrites the slot of each snapped shunt from its voltage back to the continuous slot
# value, so x can start the next solve
continuous_slots!(x, ::ControlState{Sigmoid}) = x

function continuous_slots!(x, c::ControlState{Complementarity})
    for t in c.shunts
        if t.snapped
            lo, hi = first(t.steps), last(t.steps)
            x[slot(t)] = box_slot(lo, hi, t.vset, t.scale, x[slot(t)], t.b)
        end
    end
    return x
end

## Validation, run once when the backend is built

function control_component(data, table, id)
    comp = get(get(data, table, Dict{String, Any}()), string(id), nothing)
    if comp === nothing
        throw(ArgumentError("control refers to $(table) $(id), which does not exist"))
    end
    return comp
end

function check_control(data, c::ReactiveLimits)
    gen = control_component(data, "gen", c.gen)
    if gen["gen_status"] == 0
        throw(ArgumentError("ReactiveLimits on generator $(c.gen), which is inactive"))
    end
    bus = data["bus"]["$(gen["gen_bus"])"]
    if !is_pv_bus(bus)
        throw(
            ArgumentError(
                "ReactiveLimits on generator $(c.gen) at bus $(bus["index"]), which is not " *
                    "a PV bus. The slack bus balances the network and has no limits here.",
            ),
        )
    end
    qmin, qmax = Float64(gen["qmin"]), Float64(gen["qmax"])
    if !(isfinite(qmin) && isfinite(qmax) && qmin <= qmax)
        throw(
            ArgumentError(
                "generator $(c.gen) needs finite limits with qmin <= qmax, got " *
                    "qmin = $(qmin) and qmax = $(qmax)",
            ),
        )
    end
    return nothing
end

function check_control(data, c::SwitchedShunt)
    shunt = control_component(data, "shunt", c.shunt)
    if shunt["status"] == 0
        throw(ArgumentError("SwitchedShunt on shunt $(c.shunt), which is inactive"))
    end
    bus = data["bus"]["$(shunt["shunt_bus"])"]
    if !is_pq_bus(bus["bus_type"]::Int)
        throw(
            ArgumentError(
                "SwitchedShunt on shunt $(c.shunt) at bus $(bus["index"]), which is not a " *
                    "PQ bus. A PV or slack bus already holds its own voltage.",
            ),
        )
    end
    return nothing
end

control_key(c::ReactiveLimits) = ("gen", c.gen)
control_key(c::SwitchedShunt) = ("shunt", c.shunt)

# Every active generator at a limited bus must be limited, since the bus has one setpoint
function check_limited_buses(data, controls)
    limited = Set(c.gen for c in controls if c isa ReactiveLimits)
    buses = Set(data["gen"]["$(g)"]["gen_bus"] for g in limited)
    for (id, gen) in data["gen"]
        if gen["gen_status"] != 0 && gen["gen_bus"] in buses && !(parse(Int, id) in limited)
            throw(
                ArgumentError(
                    "bus $(gen["gen_bus"]) has ReactiveLimits on some generators but not on " *
                        "generator $(id). List every active generator at the bus.",
                ),
            )
        end
    end
    return nothing
end

# The normal map gives each bus one control variable
function check_shunt_buses(data, controls, ::Complementarity)
    buses = Set{Int}()
    for c in controls
        if c isa SwitchedShunt
            bus = data["shunt"]["$(c.shunt)"]["shunt_bus"]
            if bus in buses
                throw(
                    ArgumentError(
                        "bus $(bus) has more than one SwitchedShunt, which Complementarity " *
                            "cannot solve. Merge them into one shunt or use Sigmoid.",
                    ),
                )
            end
            push!(buses, bus)
        end
    end
    return nothing
end

check_shunt_buses(data, controls, ::Sigmoid) = nothing

function check_controls(data, controls, settings::ControlSettings)
    seen = Set{Tuple{String, Int}}()
    for c in controls
        check_control(data, c)
        key = control_key(c)
        if key in seen
            throw(ArgumentError("$(key[1]) $(key[2]) has more than one control"))
        end
        push!(seen, key)
    end
    check_limited_buses(data, controls)
    check_shunt_buses(data, controls, settings.model)
    return nothing
end

## Per-state setup, run by init_state

# Removes the fixed susceptance of every switched shunt, since PowerModels folds shunts
# into the admittance matrix and the control term replaces it
function remove_switched_shunts!(work, controls)
    for c in controls
        if c isa SwitchedShunt
            work["shunt"]["$(c.shunt)"]["bs"] = 0.0
        end
    end
    return work
end

# The voltage change per unit of the control variable beyond its limits at bus row
function term_scale(pf_data, row, model::Complementarity)
    if model.scale === nothing
        return 1 / abs(imag(pf_data.am.matrix[row, row]))
    end
    return model.scale
end

# Sigmoid never leaves the voltage slot
term_scale(pf_data, row, ::Sigmoid) = 1.0

function limit_terms(work, pf_data, controls, model)
    by_bus = Dict{Int, Vector{GenShare}}()
    for c in controls
        if c isa ReactiveLimits
            gen = work["gen"]["$(c.gen)"]
            share = GenShare(c.gen, gen["qmin"], gen["qmax"])
            push!(get!(by_bus, gen["gen_bus"], GenShare[]), share)
        end
    end

    terms = LimitTerm[]
    for (busid, gens) in sort!(collect(by_bus); by = first)
        row = pf_data.am.bus_to_idx[busid]
        vset = Float64(work["bus"]["$(busid)"]["vm"])
        qmin = sum(g.qmin for g in gens)
        qmax = sum(g.qmax for g in gens)
        qg = sum(Float64(get(work["gen"]["$(g.id)"], "qg", 0.0)) for g in gens)
        qstart = clamp(qg, qmin, qmax)
        scale = term_scale(pf_data, row, model)
        sorted = sort!(gens; by = g -> g.id)
        push!(terms, LimitTerm(row, vset, qmin, qmax, qstart, scale, sorted))
    end
    return terms
end

function shunt_terms(work, pf_data, controls, model)
    terms = ShuntTerm[]
    for c in controls
        if c isa SwitchedShunt
            busid = work["shunt"]["$(c.shunt)"]["shunt_bus"]
            row = pf_data.am.bus_to_idx[busid]
            scale = term_scale(pf_data, row, model)
            term = ShuntTerm(c.shunt, row, c.vset, c.steps, scale, false, first(c.steps))
            push!(terms, term)
        end
    end
    return terms
end

# Turns each limited PV bus into a PQ bus. PowerModels has already left the generators'
# qg out of the base injection of a PV bus, so the sigmoid is their only reactive output.
# This must run before build_pf_system, whose closures read bus_type_idx.
function release_limited_buses!(pf_data, limits)
    for t in limits
        pf_data.bus_type_idx[t.row] = 1
    end
    return pf_data
end

# Under Complementarity a limited bus starts from its qg and a shunt from the middle of
# its range, both with the voltage at the setpoint
function start_slots!(x0, c::ControlState{Complementarity})
    for t in c.limits
        x0[slot(t)] = t.qstart
    end
    for t in c.shunts
        x0[slot(t)] = (first(t.steps) + last(t.steps)) / 2
    end
    return x0
end

# Under Sigmoid a limited bus starts from its setpoint rather than from PowerModels' flat
# 1.0
function start_slots!(x0, c::ControlState{Sigmoid})
    for t in c.limits
        x0[slot(t)] = t.vset
    end
    return x0
end

# The fraction of its range that every generator at a limited bus produces
function output_fraction(t::LimitTerm, c::ControlState, s)
    _, _, q, _ = slot_map(t, c, s)
    if t.qmax > t.qmin
        return (q - t.qmin) / (t.qmax - t.qmin)
    end
    return 0.0
end

## Solving

function solve_from(state, b, start)
    copyto!(state.sys.x0, start)
    return PM._solve_nl!(state.cache, state.sys, b.alg; b.solver_kwargs...)
end

function solve_continuous(state, b, start, ::Complementarity)
    sol = solve_from(state, b, start)
    return sol, sol.iterations
end

# Solves at the target steepness. If that fails, walks the homotopy steepnesses up to the
# target, each solve starting from the previous solution.
function solve_continuous(state, b, start, model::Sigmoid)
    c = state.controls
    c.steepness = model.steepness
    sol = solve_from(state, b, start)
    iterations = sol.iterations
    if sol.converged || isempty(model.homotopy)
        return sol, iterations
    end

    x = start
    for steepness in vcat(model.homotopy, model.steepness)
        c.steepness = steepness
        sol = solve_from(state, b, x)
        iterations += sol.iterations
        if !sol.converged
            return sol, iterations
        end
        x = copy(sol.x)
    end
    return sol, iterations
end

# The continuous solve, then a second solve with every switched shunt snapped to a step
function solve_controlled(state, b, start)
    c = state.controls
    unsnap!(c)
    sol, iterations = solve_continuous(state, b, start, c.model)
    if sol.converged && b.control_settings.snap && !isempty(c.shunts)
        sol = solve_from(state, b, snap!(c, copy(sol.x)))
        iterations += sol.iterations
    end
    return sol, iterations
end
