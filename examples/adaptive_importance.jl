# Probability that a voltage leaves a band on both sides, by adaptive importance sampling.
#
# In PowerModels' case5, bus 2 is the only PQ bus, so its voltage is the one that varies.
# The loads at buses 2 and 3 are Normal and coupled by a Gaussian copula. We estimate the
# probability that the voltage at bus 2 leaves a band around its median. High load gives
# an undervoltage and low load an overvoltage, so the violations lie in two separate
# regions of the input space.
#
# DesignPointSampling shifts its samples toward the nearest region only. It misses the
# other one and reports a standard error that does not show it. AdaptiveDesignPointSampling
# finds a design point for each bound and adapts a proposal around each.
#
# Run from the repository root:
#
#     julia --project=examples examples/adaptive_importance.jl

using Pkg
Pkg.activate(joinpath(@__DIR__))

using ProbabilisticPowerFlow
using ImportanceSamplers: ImportanceSamplers
using PowerModels: PowerModels
using Distributions: Normal
using Copulas: GaussianCopula
using Random: Xoshiro
using Statistics: median, std

PowerModels.silence()

case = joinpath(pkgdir(PowerModels), "test", "data", "matpower", "case5.m")
data = PowerModels.parse_file(case)

load_at(bus) = parse(Int, only(id for (id, l) in data["load"] if l["load_bus"] == bus))
pd(bus) = data["load"][string(load_at(bus))]["pd"]

buses = [2, 3]
variables = [GermVariable("pd$(b)", Normal(pd(b), 0.1 * pd(b))) for b in buses]
assignments = [Assignment("pd$(b)", ComponentRef(ComponentField.Pd, load_at(b))) for b in buses]
model = UncertaintyModel(variables, assignments, GaussianCopula([1.0 0.5; 0.5 1.0]))

vm = VoltageMagnitude(2)
backend = PowerModelsBackend(data)

# a band of 3.6 standard deviations on either side of the median voltage
reference = solve(PPFProblem(backend, model, [vm]), LatinHypercube(n = 400); rng = Xoshiro(0))
center = median(qoi_samples(reference, vm))
spread = std(reference, vm)
band = ViolationEvent(vm, center - 3.6 * spread, center + 3.6 * spread)
prob = PPFProblem(backend, model, [vm, band])

function report(name, result)
    p = violation_probability(result, band)
    se = standard_error(result, band)
    println(
        rpad(name, 32), "P = $(round(p; sigdigits = 3)), standard error ",
        "$(round(se; sigdigits = 2)), $(result.n_solves) power flows",
    )
    return nothing
end

# both importance samplers at about 3000 power flows
plain = solve(prob, DesignPointSampling(band; n = 3000); rng = Xoshiro(1))
adaptive = solve(prob, AdaptiveDesignPointSampling(band; n = 1000); rng = Xoshiro(1))
display(adaptive)
println()

# plain Monte Carlo as the reference, which takes about a minute
mc = solve(prob, MonteCarlo(n = 1_000_000); rng = Xoshiro(2))

println("band [$(round(band.lo, digits = 4)), $(round(band.hi, digits = 4))] pu")
report("DesignPointSampling", plain)
report("AdaptiveDesignPointSampling", adaptive)
report("Monte Carlo", mc)
println("design point of DesignPointSampling: ", round.(plain.method.shift; digits = 2))
println("design points of AdaptiveDesignPointSampling: ", [round.(s; digits = 2) for s in adaptive.method.shift])
