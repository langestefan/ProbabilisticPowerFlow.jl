# End-to-end probabilistic power flow on the IEEE 14-bus case.
#
#   julia --project=examples -t auto examples/demo.jl
#
# One load level and one wind speed, correlated through a Gaussian copula, drive the
# injections. Monte Carlo turns them into a distribution of bus voltages and a
# violation probability.

using ProbabilisticPowerFlow
using ProbabilisticPowerFlow: AffineTransform
using PowerModels
using Distributions
using Copulas
using Random: Xoshiro
using Statistics: mean, std, quantile
using Printf

PowerModels.silence()

const CASE = joinpath(@__DIR__, "data", "case14.m")

# case14 has its reference at bus 1 and PV buses at 2, 3, 6 and 8.
const PV_BUSES = Set([2, 3, 6, 8])

# The base case is comfortably within limits, so nothing interesting happens at
# nominal load. STRESS scales every load to put the network near where voltage
# problems start.
const STRESS = 1.5
const LO, HI = 0.95, 1.05

data = PowerModels.parse_file(CASE)
backend = PowerModelsBackend(data)

# ---------------------------------------------------------------------------
# The uncertainty.
#
# The germ is the only random thing in the whole run. Everything after it is a
# deterministic function of it, which is why a sample can be replayed from its u
# point alone, as the cross-check at the bottom does.
# ---------------------------------------------------------------------------

variables = [
    GermVariable("load_level", Normal(1.0, 0.25)),   # 1.0 is the stressed base case
    GermVariable("wind_speed", Weibull(2.0, 8.0)),   # m/s at the farm
    GermVariable("unit_2", Normal(0.40, 0.05)),      # pu setpoint of generator 2
]

# One load level scales all eleven loads at once. Each load gets its own
# AffineTransform carrying its own nominal value, so the network breathes together
# and every load keeps its power factor. That shared germ variable is what makes the
# loads perfectly correlated with each other.
assignments = Assignment[]
for (id, load) in sort(collect(data["load"]), by = kv -> parse(Int, kv[1]))
    i = parse(Int, id)
    push!(
        assignments,
        Assignment(
            "load_level",
            ComponentRef(ComponentField.Pd, i),
            AffineTransform(STRESS * load["pd"], 0.0),
        ),
    )
    # Reactive load at a PV bus is absorbed by the voltage setpoint, so assigning it
    # would be a value silently thrown away. init_state rejects it rather than let
    # that happen, so only the PQ-bus loads get a Qd assignment.
    load["load_bus"] in PV_BUSES && continue
    push!(
        assignments,
        Assignment(
            "load_level",
            ComponentRef(ComponentField.Qd, i),
            AffineTransform(STRESS * load["qd"], 0.0),
        ),
    )
end

# A wind farm at PQ bus 5, which is load 4. PowerModels represents generators only at
# PV and reference buses, so an injection at a PQ bus is written as a negative load:
# 0.10 pu per m/s of wind, offset by that bus's own load.
push!(
    assignments,
    Assignment(
        "wind_speed",
        ComponentRef(ComponentField.Pd, 4),
        AffineTransform(-0.10, STRESS * data["load"]["4"]["pd"]),
    ),
)

# A dispatchable unit at PV bus 2, uncertain around its schedule.
push!(assignments, Assignment("unit_2", ComponentRef(ComponentField.Pg, 2)))

# Windy days here are mild ones, so demand runs lower when the farm produces more.
# The unit's schedule is independent of the weather.
correlation = [
    1.0 -0.5 0.0
    -0.5 1.0 0.0
    0.0 0.0 1.0
]

model = UncertaintyModel(variables, assignments, GaussianCopula(correlation))

# ---------------------------------------------------------------------------
# What to measure. Bus 14 is the far end of the network and the first to sag.
# ---------------------------------------------------------------------------

const WEAK = 14

qois = AbstractQoI[
    VoltageMagnitude(WEAK),
    VoltageMagnitude(4),
    VoltageMagnitude(5),
    BranchActivePower(4, 5),
    ViolationEvent(VoltageMagnitude(WEAK), LO, HI),
]

problem = PPFProblem(backend, model, qois)

println(repeat("=", 72))
display(problem)
println("\n")

# ---------------------------------------------------------------------------
# Run it.
# ---------------------------------------------------------------------------

method = MonteCarlo(n = 5000, warmstart = :sorted, keep_inputs = true)
result = solve(problem, method; rng = Xoshiro(20260902), ntasks = Threads.nthreads())

println(repeat("=", 72))
display(result)
println("\n")

# ---------------------------------------------------------------------------
# Read it.
# ---------------------------------------------------------------------------

@printf("%-24s %8s %8s %8s %8s\n", "quantity", "mean", "std", "p5", "p95")
println(repeat("-", 62))
for q in qois[1:4]
    @printf(
        "%-24s %8.4f %8.4f %8.4f %8.4f\n",
        sprint(show, q),
        mean(result, q),
        std(result, q),
        quantile(result, q, 0.05),
        quantile(result, q, 0.95),
    )
end

event = qois[5]
p = violation_probability(result, event)
n = n_converged(result)
# an indicator is Bernoulli, so the mean of n of them has standard error √(p(1-p)/n)
se = sqrt(p * (1 - p) / n)

println()
@printf(
    "P(vm at bus %d outside [%.2f, %.2f]) = %.4f ± %.4f  (%.0f%% relative)\n",
    WEAK,
    LO,
    HI,
    p,
    se,
    100 * se / p
)
@printf(
    "  the σ/√n rule puts 10%% relative error at about %d samples\n",
    round(Int, 100 / p)
)

# Which side of the band is doing the work. Both tails are in the same indicator, so
# splitting them needs the underlying quantity, which is why it was recorded too.
vm = qoi_samples(result, VoltageMagnitude(WEAK))
@printf(
    "  below %.2f: %.4f    above %.2f: %.4f\n",
    LO,
    count(<(LO), vm) / n,
    HI,
    count(>(HI), vm) / n
)

# Any other band on a recorded quantity costs nothing: the indicator is a
# deterministic function of samples already in hand, so no new solves are needed.
println("\nOther bands on the same 5000 solves:")
for lo in (0.93, 0.94, 0.95, 0.96)
    q = violation_probability(result, ViolationEvent(VoltageMagnitude(WEAK), lo, HI))
    @printf("  [%.2f, %.2f] -> %.4f\n", lo, HI, q)
end

# ---------------------------------------------------------------------------
# Replay. The pipeline is deterministic given u, so a stored sample can be pushed
# through the backend by hand and must reproduce its recorded value exactly.
# ---------------------------------------------------------------------------

j = 1
u = result.u[:, j]
x = to_physical(model, u)
state = init_state(backend, targets(model))
set_injections!(state, backend, x)
info = solve!(state, backend)

println("\n", repeat("=", 72))
println("Replaying stored sample $(result.sample_indices[j])")
@printf("  u             = [%s]\n", join((@sprintf("%.4f", v) for v in u), ", "))
@printf("  injections[1:4] = [%s]\n", join((@sprintf("%.4f", v) for v in x[1:4]), ", "))
@printf("  %s\n", sprint(show, info))
@printf(
    "  vm(%d) replayed %.12f vs recorded %.12f\n",
    WEAK,
    extract(state, backend, VoltageMagnitude(WEAK)),
    result.samples[1, j],
)

# ---------------------------------------------------------------------------
# Divergence. Pushed far enough past its loadability limit the case stops having a
# solution. Those samples are recorded, not dropped and not thrown: where the solver
# fails is itself information about the feasibility boundary.
# ---------------------------------------------------------------------------

hard = UncertaintyModel(
    [GermVariable("load_level", Normal(2.0, 0.4)), variables[2], variables[3]],
    assignments,
    GaussianCopula(correlation),
)
hard_result =
    solve(PPFProblem(backend, hard, qois), MonteCarlo(n = 500); rng = Xoshiro(20260902))

println("\n", repeat("=", 72))
@printf(
    "Past the loadability limit: %d of %d solves diverged (%.1f%%)\n",
    length(hard_result.failures),
    hard_result.n_samples,
    100 * failure_rate(hard_result),
)
if !isempty(hard_result.failures)
    f = first(hard_result.failures)
    @printf(
        "  first failure was sample %d, total injection %.3f pu, %s\n",
        f.index,
        sum(f.injections),
        sprint(show, f.info),
    )
end
