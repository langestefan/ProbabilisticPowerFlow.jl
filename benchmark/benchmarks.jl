# Performance benchmarks for ProbabilisticPowerFlow.jl, compatible with AirspeedVelocity.jl.
#
# Compare the working tree against main from the repository root:
#
#     benchpkg ProbabilisticPowerFlow --path=. --rev=main,dirty
#
# Or run the suite directly:
#
#     julia --project=benchmark -e 'include("benchmark/benchmarks.jl"); run(SUITE; verbose = true)'

using BenchmarkTools
using ProbabilisticPowerFlow
using PowerModels: PowerModels
using NonlinearSolve: NewtonRaphson
using LinearSolve: KLUFactorization
using OhMyThreads: DynamicScheduler
using QuasiMonteCarlo: SobolSample, HaltonSample, LatticeRuleSample, OwenScramble
using Distributions: Normal
using Random: Xoshiro

PowerModels.silence()

const SUITE = BenchmarkGroup()

# MATPOWER cases shipped with PowerModels
const CASES = ["case14", "case30"]

const SOLVERS = Dict(
    "NativeNewton" => PowerModels.NativeNewton(),
    "NewtonRaphson" => NewtonRaphson(linsolve = KLUFactorization(check_pattern = false)),
)

# a power of 2, which Sobol points need to keep their balance
const N_SAMPLES = 512

const METHODS = [
    "MonteCarlo" => MonteCarlo(n = N_SAMPLES),
    "LatinHypercube" => LatinHypercube(n = N_SAMPLES),
    "QuasiMC, SobolSample" => QuasiMC(SobolSample(); n = N_SAMPLES),
    "QuasiMC, SobolSample, OwenScramble" =>
        QuasiMC(SobolSample(R = OwenScramble(base = 2, pad = 32)); n = N_SAMPLES),
    "QuasiMC, HaltonSample" => QuasiMC(HaltonSample(); n = N_SAMPLES),
    "QuasiMC, LatticeRuleSample" => QuasiMC(LatticeRuleSample(); n = N_SAMPLES),
]

# QuasiMC draws its points from the sampler, so only the random methods take an rng
sample_kwargs(::QuasiMC) = (;)
sample_kwargs(::AbstractPPFMethod) = (rng = Xoshiro(1),)

case_data(name) = PowerModels.parse_file(
    joinpath(pkgdir(PowerModels), "test", "data", "matpower", "$name.m"),
)

base_pd(data, id) = data["load"][string(id)]["pd"]

is_slack_bus(data, bus) = data["bus"][string(bus)]["bus_type"] == 3

# active loads with a positive demand away from the slack bus, which the backend can assign
function assignable_loads(data)
    ids = Int[]
    for (id, load) in data["load"]
        if load["status"] != 0 && load["pd"] > 0 && !is_slack_bus(data, load["load_bus"])
            push!(ids, parse(Int, id))
        end
    end
    return sort!(ids)
end

# every assignable load varies independently by 10 percent around its base value
function load_model(data)
    ids = assignable_loads(data)
    variables = [GermVariable("pd$id", Normal(base_pd(data, id), 0.1 * base_pd(data, id))) for id in ids]
    assignments = [Assignment("pd$id", ComponentRef(ComponentField.Pd, id)) for id in ids]
    return UncertaintyModel(variables, assignments)
end

voltage_qois(data) = [VoltageMagnitude(bus["index"]) for bus in values(data["bus"])]

for case in CASES
    data = case_data(case)
    model = load_model(data)
    refs = targets(model)
    x = [base_pd(data, ref.id) for ref in refs]
    qois = voltage_qois(data)

    SUITE["backend"][case]["construct"] = @benchmarkable PowerModelsBackend($data)

    for (name, alg) in SOLVERS
        backend = PowerModelsBackend(data; alg)
        state = init_state(backend, refs)
        set_injections!(state, backend, x)
        prob = PPFProblem(backend, model, qois)

        # a single deterministic solve at base injections, from a cold start
        SUITE["backend"][case]["init_state"][name] = @benchmarkable init_state($backend, $refs)
        SUITE["backend"][case]["solve!"][name] = @benchmarkable solve!($state, $backend)

        sampling = SUITE["sampling"][case][name]["n=$N_SAMPLES"]
        for (label, method) in METHODS
            sampling[label] = @benchmarkable solve($prob, $method; sample_kwargs($method)...)
        end
        sampling["MonteCarlo, warmstart = :chain"] = @benchmarkable solve(
            $prob, MonteCarlo(n = N_SAMPLES); rng = Xoshiro(1), warmstart = :chain,
        )
        sampling["MonteCarlo, warmstart = :sorted"] = @benchmarkable solve(
            $prob, MonteCarlo(n = N_SAMPLES); rng = Xoshiro(1), warmstart = :sorted,
        )
        sampling["MonteCarlo, DynamicScheduler"] = @benchmarkable solve(
            $prob, MonteCarlo(n = N_SAMPLES); rng = Xoshiro(1), scheduler = DynamicScheduler(),
        )
    end
end
