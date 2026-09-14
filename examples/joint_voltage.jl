# Joint distribution of the voltage magnitude and angle at a bus.
#
# Run from the repository root:
#
#     julia --project=examples examples/joint_voltage.jl

using Pkg
println("Activating project environment in $(pwd())")
Pkg.activate(joinpath(@__DIR__))

using ProbabilisticPowerFlow
using PowerModels: PowerModels
using NonlinearSolve: NewtonRaphson
using LinearSolve: KLUFactorization
using Distributions: Normal
using Copulas: GaussianCopula
using Random: Xoshiro
using Statistics: cor, mean, std

PowerModels.silence()

const BUS = 2

# the id of the single load at a bus
load_id(data, bus) = parse(Int, only(id for (id, l) in data["load"] if l["load_bus"] == bus))

case = joinpath(pkgdir(PowerModels), "test", "data", "matpower", "case5.m")
data = PowerModels.parse_file(case)

id = load_id(data, BUS)
pd = data["load"][string(id)]["pd"]
qd = data["load"][string(id)]["qd"]

# both loads vary by 10 percent around their base value, with correlation 0.8
variables = [
    GermVariable("pd", Normal(pd, 0.1 * pd)),
    GermVariable("qd", Normal(qd, 0.1 * qd)),
]
assignments = [
    Assignment("pd", ComponentRef(ComponentField.Pd, id)),
    Assignment("qd", ComponentRef(ComponentField.Qd, id)),
]
model = UncertaintyModel(variables, assignments, GaussianCopula([1.0 0.8; 0.8 1.0]))

vm = VoltageMagnitude(BUS)
va = VoltageAngle(BUS)

# a NonlinearSolve algorithm with a sparse KLU factorization
backend = PowerModelsBackend(
    data;
    alg = NewtonRaphson(linsolve = KLUFactorization(check_pattern = false)),
    solver_kwargs = (abstol = 1.0e-10, maxiters = 20),
)
prob = PPFProblem(backend, model, [vm, va])

# first try plain Monte Carlo sampling, which is the default
result_mc = solve(prob, MonteCarlo(n = 1000); rng = Xoshiro(1))

# try again with Latin Hypercube sampling, which is more efficient than plain Monte Carlo sampling
result_lhs = solve(prob, LatinHypercube(n = 300); rng = Xoshiro(1))

joint = result_lhs.samples

display(result_lhs)

println()
println("vm mean $(mean(result_lhs, vm)) pu, std $(std(result_lhs, vm))")
println("va mean $(mean(result_lhs, va)) rad, std $(std(result_lhs, va))")
println("correlation $(cor(qoi_samples(result_lhs, vm), qoi_samples(result_lhs, va)))")
