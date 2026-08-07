"""
Benchmarks for ProbabilisticPowerFlow.jl

Compatible with AirspeedVelocity.jl for benchmarking across package versions.

Run locally with:

    benchpkg ProbabilisticPowerFlow --rev=main,dirty

Or use the Julia API:

    using BenchmarkTools
    include("benchmark/benchmarks.jl")
    run(SUITE)

The statistical convergence study of the sampling methods is a separate script,
see `convergence.jl` in this folder.
"""

using BenchmarkTools
using ProbabilisticPowerFlow
using Distributions: Normal
using Random: Xoshiro

const SUITE = BenchmarkGroup()

# Normal load uncertainty on the three PQ buses of case5, loads at buses 3 and 4
# rank-correlated. The same setup as the test suite uses.
function case5_problem()
    base_pd = [0.45, 0.40, 0.60]
    vars = [
        GermVariable("load$(bus)", Normal(base_pd[k], 0.1 * base_pd[k])) for
        (k, bus) in enumerate(3:5)
    ]
    assigns = [Assignment("load$(bus)", ComponentRef(:load, "$(bus)", :pd)) for bus = 3:5]
    R = [1.0 0.5 0.0; 0.5 1.0 0.0; 0.0 0.0 1.0]
    model = UncertaintyModel(vars, assigns, GaussianCopula(R))
    qois = [VoltageMagnitude(5)]
    return PPFProblem(ReferenceBackend(case5()), model, qois)
end

const PROB = case5_problem()

# the u -> germ -> injections pipeline without any power flow solve
SUITE["pipeline"]["to_physical"] =
    @benchmarkable to_physical(PROB.model, u) setup = (u = rand(Xoshiro(1), 3))

# one deterministic solve through the backend contract
SUITE["backend"]["cold_solve"] = @benchmarkable solve!(state, PROB.backend) setup =
    (state = init_state(PROB.backend, ComponentRef[]))

# sampling methods, 100 samples each, identical uncertainty model
for (name, method) in (
    ("MonteCarlo", MonteCarlo(n = 100)),
    ("MonteCarlo_chain", MonteCarlo(n = 100, warmstart = :chain)),
    ("MonteCarlo_sorted", MonteCarlo(n = 100, warmstart = :sorted)),
    ("LatinHypercube", LatinHypercube(n = 100)),
)
    SUITE["methods"][name] =
        @benchmarkable solve(PROB, $method; rng = Xoshiro(1)) seconds = 10
end
