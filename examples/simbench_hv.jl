# Probabilistic power flow on a SimBench high-voltage benchmark grid.
#
#   julia --project=examples -t auto examples/simbench_hv.jl
#
# Unlike examples/demo.jl, which invents its distributions, everything uncertain here
# comes out of SimBench's own data: the marginals are the empirical distributions of
# its year of quarter-hourly profiles, and the copula's correlation is estimated from
# the same series. The dataset downloads itself on first use, about 88 MB.

using ProbabilisticPowerFlow
using ProbabilisticPowerFlow: AffineTransform
using SimBench
using PowerModels
using DataFrames
using Distributions
using Copulas
using LinearAlgebra: diagind, isposdef
using Random: Xoshiro
using Statistics: mean, std, quantile, cor
using Printf

PowerModels.silence()

# SimBench's standalone HV grids, by electrical bus count:
#
#   1-HV-mixed--0-no_sw    64 buses   161 elements   this script
#   1-HV-urban--0-no_sw    82 buses   177 elements
#   1-HV-mixed--0-sw      306 buses   the same grid with switch buses kept
#   1-HV-urban--0-sw      372 buses
#
# The no_sw variants collapse switch buses into the electrical topology. They solve in
# three Newton iterations where 1-HV-mixed--0-sw takes thirty, which matters when the
# same network is solved thousands of times.
const GRID = "1-HV-mixed--0-no_sw"

# The German HV planning band.
const LO, HI = 0.90, 1.10

# ---------------------------------------------------------------------------
# The network.
# ---------------------------------------------------------------------------

grid = SimBench.read_grid(GRID)
data = SimBench.powermodels_data(grid)

# The nominal powers, before the study case scales them. A profile is relative to the
# element's nominal rating, so these are the AffineTransform scales below.
nominal = Dict(id => (load["pd"], load["qd"]) for (id, load) in data["load"])

# The reference voltage matters more than anything else here. Straight out of the CSV
# the three boundary nodes carry 1.092, 1.092 and 1.068, which are the setpoints of the
# EHV grid they are cut from, and running with them sits the whole study about six
# percent too high. Every one of SimBench's six study cases puts the slack at 1.025
# instead, so the choice of case only affects powers, which the profiles then replace.
SimBench.apply_study_case!(data, grid, "hW")

backend = PowerModelsBackend(data)

# SimBench's RES units become PowerModels loads with a negative pd, so generation and
# demand are the same kind of assignment here and differ only in the sign already
# carried by the element's nominal value.
res_units = count(l -> l["source_id"][1] == "sgen", values(data["load"]))

@printf(
    "%s: %d buses, %d branches, %d loads, %d RES units, %d reference buses\n",
    GRID,
    length(data["bus"]),
    length(data["branch"]),
    length(data["load"]) - res_units,
    res_units,
    count(b -> b["bus_type"] == 3, values(data["bus"])),
)

# ---------------------------------------------------------------------------
# The germ, read off the profiles.
#
# A SimBench profile is a relative series in [0, 1] that scales an element's nominal
# power, so one germ variable per profile is exactly the right granularity: every
# element following that profile moves with it, which is what an assignment expresses.
# ---------------------------------------------------------------------------

# The empirical distribution of a profile, with no distributional assumption. Wind is
# skewed and load is not, and neither is anything a Normal would describe; using the
# data as its own marginal sidesteps the question. Any UnivariateDistribution works
# here, because to_physical! only ever calls quantile on it.
function empirical(x::AbstractVector)
    v = collect(skipmissing(x))
    support = sort(unique(v))
    counts = Dict{Float64,Int}()
    for y in v
        counts[y] = get(counts, y, 0) + 1
    end
    return DiscreteNonParametric(support, [counts[s] / length(v) for s in support])
end

load_profiles = sort(unique(skipmissing(grid[:Load].profile)))
res_profiles = sort(unique(skipmissing(grid[:RES].profile)))

# SimBench gives a load two profiles, one for p and one for q, and they are not the
# same shape: their correlation across the year is about 0.8, not 1. So each gets its
# own germ variable and the copula carries the dependence between them, rather than
# the model asserting a constant power factor that the data does not support.
series = Pair{String,Vector{Float64}}[]
for t in load_profiles
    push!(series, "load:$(t):p" => collect(skipmissing(grid[:LoadProfile][!, t*"_pload"])))
    push!(series, "load:$(t):q" => collect(skipmissing(grid[:LoadProfile][!, t*"_qload"])))
end
for t in res_profiles
    push!(series, "res:$(t)" => collect(skipmissing(grid[:RESProfile][!, t])))
end

# ---------------------------------------------------------------------------
# The assignments, one per element power that is actually nonzero.
# ---------------------------------------------------------------------------

profile_of = Dict{String,String}()
for row in eachrow(grid[:Load])
    profile_of[row.id] = "load:$(row.profile)"
end
for row in eachrow(grid[:RES])
    profile_of[row.id] = "res:$(row.profile)"
end

assignments = Assignment[]
for (id, load) in sort(collect(data["load"]), by = kv -> parse(Int, kv[1]))
    j = parse(Int, id)
    base = profile_of[load["name"]]
    kind = load["source_id"][1]
    pd, qd = nominal[id]

    # A load's q follows its own profile; a RES unit has only one profile, so its q
    # moves with its p exactly, which is the shared-germ case.
    p_var = kind == "load" ? base * ":p" : base
    q_var = kind == "load" ? base * ":q" : base

    # An element with no nominal power contributes an assignment whose transform is
    # identically zero. Dropping it keeps the germ honest: a variable no assignment
    # reads is rejected by UncertaintyModel, which is how the 46 zero-rated RES units
    # in this grid remove their profile from the model rather than pad it.
    if pd != 0
        push!(
            assignments,
            Assignment(p_var, ComponentRef(ComponentField.Pd, j), AffineTransform(pd, 0.0)),
        )
    end
    if qd != 0
        push!(
            assignments,
            Assignment(q_var, ComponentRef(ComponentField.Qd, j), AffineTransform(qd, 0.0)),
        )
    end
end

# Keep only the profiles something actually reads.
used = Set(a.variable for a in assignments)
series = [kv for kv in series if first(kv) in used]
variables = [GermVariable(k, empirical(v)) for (k, v) in series]

# ---------------------------------------------------------------------------
# The dependence, estimated from the same series.
#
# Transforming each series to normal scores and taking their correlation is the
# semiparametric estimator of a Gaussian copula: it is invariant to the marginals,
# which is the whole point of separating them from the dependence.
# ---------------------------------------------------------------------------

function normal_scores(x::AbstractVector)
    n = length(x)
    ranks = invperm(sortperm(x))
    return quantile.(Normal(), ranks ./ (n + 1))
end

Z = reduce(hcat, normal_scores(v) for (_, v) in series)
R = cor(Z)
R = (R + R') / 2                      # symmetric up to floating point
R[diagind(R)] .= 1.0
@assert isposdef(R)

model = UncertaintyModel(variables, assignments, GaussianCopula(R))

@printf(
    "\ngerm dimension %d, %d assignments, estimated from %d quarter-hourly steps\n",
    germ_dim(model),
    length(model.assignments),
    size(Z, 1),
)
for (k, v) in enumerate(model.variables)
    @printf("  %-18s mean %.3f  p95 %.3f\n", v.id, mean(v.dist), quantile(v.dist, 0.95))
end

# ---------------------------------------------------------------------------
# What to measure: every bus, so the weakest one can be found afterwards rather
# than guessed at.
# ---------------------------------------------------------------------------

buses = sort(parse.(Int, collect(keys(data["bus"]))))
qois = AbstractQoI[VoltageMagnitude(b) for b in buses]

# Line loading is the real question on this grid, since it carries twice as much wind
# capacity as load. A flow QoI names its branch by the two buses it runs between, which
# cannot pick one of two parallel circuits, and double-circuit lines are the norm at
# this voltage: 70 of the 101 branches here share a bus pair with another. Those are
# skipped, and counted below.
lines = Dict{Tuple{Int,Int},Float64}()
ambiguous = Set{Tuple{Int,Int}}()
for br in values(data["branch"])
    br["br_status"] == 0 && continue
    key = (br["f_bus"], br["t_bus"])
    if haskey(lines, key) || haskey(lines, reverse(key))
        push!(ambiguous, key)
    else
        lines[key] = br["rate_a"]
    end
end
for key in ambiguous
    delete!(lines, key)
    delete!(lines, reverse(key))
end
addressable = sort(collect(keys(lines)))
filter!(k -> lines[k] > 0, addressable)

for (f, t) in addressable
    push!(qois, BranchActivePower(f, t))
    push!(qois, BranchReactivePower(f, t))
end

problem = PPFProblem(backend, model, qois)

method = MonteCarlo(n = 5000, warmstart = :sorted, keep_inputs = true)
elapsed = @elapsed result =
    solve(problem, method; rng = Xoshiro(20260902), ntasks = Threads.nthreads())

println()
display(result)
@printf(
    "\n\n%d solves in %.1f s on %d threads, %.1f ms per solve\n",
    result.n_solves,
    elapsed,
    Threads.nthreads(),
    1000 * elapsed / result.n_solves,
)

# ---------------------------------------------------------------------------
# Read it.
# ---------------------------------------------------------------------------

spread(b) =
    quantile(result, VoltageMagnitude(b), 0.95) -
    quantile(result, VoltageMagnitude(b), 0.05)
ranked = sort(buses; by = spread, rev = true)

println("\nWidest voltage spread (per unit):")
@printf("%-8s %8s %8s %8s %8s %8s\n", "bus", "mean", "std", "p5", "p95", "min")
println(repeat("-", 52))
for b in ranked[1:8]
    x = qoi_samples(result, VoltageMagnitude(b))
    @printf(
        "%-8d %8.4f %8.4f %8.4f %8.4f %8.4f\n",
        b,
        mean(result, VoltageMagnitude(b)),
        std(result, VoltageMagnitude(b)),
        quantile(result, VoltageMagnitude(b), 0.05),
        quantile(result, VoltageMagnitude(b), 0.95),
        minimum(x),
    )
end

# Every band on every recorded bus is available after the fact, because a
# ViolationEvent is a deterministic function of samples already in hand.
println("\nViolation probability outside [$(LO), $(HI)], by bus:")
n = n_converged(result)
worst = sort(
    buses;
    by = b ->
        violation_probability(result, ViolationEvent(VoltageMagnitude(b), LO, HI)),
    rev = true,
)
breached = [
    (b, violation_probability(result, ViolationEvent(VoltageMagnitude(b), LO, HI))) for
    b in worst[1:5]
]
filter!(t -> last(t) > 0, breached)
if isempty(breached)
    println("  none: the band is never left in $(n) samples")
else
    for (b, p) in breached
        @printf("  bus %-6d %.4f ± %.4f\n", b, p, sqrt(p * (1 - p) / n))
    end
end

# The grid-wide extremes and the headroom left, which is what a planner asks for when
# the answer to "does it break" is no.
lo_all = minimum(minimum(qoi_samples(result, VoltageMagnitude(b))) for b in buses)
hi_all = maximum(maximum(qoi_samples(result, VoltageMagnitude(b))) for b in buses)
@printf(
    "\nAcross all %d buses and %d samples: vm ∈ [%.4f, %.4f]\n",
    length(buses),
    n,
    lo_all,
    hi_all,
)
@printf("  headroom: %.4f above %.2f, %.4f below %.2f\n", lo_all - LO, LO, HI - hi_all, HI)

# ---------------------------------------------------------------------------
# Line loading. Apparent power is not a QoI of its own, so it is reconstructed from
# the active and reactive samples after the run. That works because a QoI sample is
# just a number in a matrix, but it does mean the loading cannot be written as a
# ViolationEvent and estimated directly, which a combined QoI would fix.
# ---------------------------------------------------------------------------

println(
    "\nLine loading, |S| / rate_a, on the $(length(addressable)) of " *
    "$(count(b -> b["br_status"] != 0, values(data["branch"]))) branches a bus pair " *
    "can address:",
)

loading = map(addressable) do (f, t)
    pf = qoi_samples(result, BranchActivePower(f, t))
    qf = qoi_samples(result, BranchReactivePower(f, t))
    return (f, t), hypot.(pf, qf) ./ lines[(f, t)]
end
sort!(loading; by = kv -> maximum(last(kv)), rev = true)

@printf("%-14s %8s %8s %8s %8s\n", "branch", "mean", "p95", "max", "P(>1)")
println(repeat("-", 50))
for ((f, t), x) in first(loading, 6)
    @printf(
        "%-14s %8.3f %8.3f %8.3f %8.4f\n",
        "$(f) → $(t)",
        mean(x),
        quantile(x, 0.95),
        maximum(x),
        count(>(1.0), x) / length(x),
    )
end

worst_line, worst_x = first(loading)
@printf(
    "\nMost loaded addressable circuit is %d → %d, peaking at %.1f%% of its rating.\n",
    worst_line[1],
    worst_line[2],
    100 * maximum(worst_x),
)
