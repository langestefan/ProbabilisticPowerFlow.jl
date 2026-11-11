# Probability of a rare overvoltage, by importance sampling.
#
# In PowerModels' case5, bus 2 is the only PQ bus, so its voltage is the one that varies.
# The load there is Normal, wind at bus 3 follows a Weibull distribution and offsets the
# load at that bus, and a Gaussian copula couples the two. We estimate the probability
# that the voltage at bus 2 exceeds a threshold that plain Monte Carlo hits only rarely.
#
# Run from the repository root:
#
#     julia --project=examples examples/rare_overvoltage.jl

using Pkg
Pkg.activate(joinpath(@__DIR__))

using ProbabilisticPowerFlow
using PowerModels: PowerModels
using Distributions: Normal, Weibull
using Copulas: GaussianCopula
using Random: Xoshiro
using Statistics: quantile, std

PowerModels.silence()

case = joinpath(pkgdir(PowerModels), "test", "data", "matpower", "case5.m")
data = PowerModels.parse_file(case)

load_at(bus) = parse(Int, only(id for (id, l) in data["load"] if l["load_bus"] == bus))
pd(bus) = data["load"][string(load_at(bus))]["pd"]

variables = [
    GermVariable("pd2", Normal(pd(2), 0.1 * pd(2))),
    GermVariable("wind3", Weibull(2.0, 8.0)),
]
assignments = [
    Assignment("pd2", ComponentRef(ComponentField.Pd, load_at(2))),
    # net load at bus 3 drops by 5% of its base value per m/s of wind
    Assignment(
        "wind3",
        ComponentRef(ComponentField.Pd, load_at(3)),
        AffineTransform(-0.05 * pd(3), pd(3)),
    ),
]
model = UncertaintyModel(variables, assignments, GaussianCopula([1.0 -0.4; -0.4 1.0]))

vm = VoltageMagnitude(2)
backend = PowerModelsBackend(data)

# put the threshold well above the median voltage, so violations are rare
reference = solve(PPFProblem(backend, model, [vm]), LatinHypercube(n = 200); rng = Xoshiro(0))
threshold = quantile(reference, vm, 0.5) + 4 * std(reference, vm)
over = ViolationEvent(vm, -Inf, threshold)
prob = PPFProblem(backend, model, [vm, over])

is = solve(prob, DesignPointSampling(over; n = 2000); rng = Xoshiro(1))
display(is)
println()

p = violation_probability(is, over)
cov = standard_error(is, over) / p
println("threshold $(round(threshold, digits = 4)) pu")
println("importance sampling: P = $(p), coefficient of variation $(round(cov, digits = 3))")

# plain Monte Carlo needs far more samples for the same precision
mc = solve(prob, MonteCarlo(n = 50_000); rng = Xoshiro(1))
p_mc = violation_probability(mc, over)
cov_mc = standard_error(mc, over) / p_mc
println("Monte Carlo:         P = $(p_mc), coefficient of variation $(round(cov_mc, digits = 3))")
