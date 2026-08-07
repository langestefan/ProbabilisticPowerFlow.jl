"""
Statistical convergence of the sampling methods on the 1354-bus PEGASE case.

Estimates the mean voltage magnitude at the bus most sensitive to load
uncertainty, for growing sample counts and several replicates per method, and
reports the root mean square error against a high-accuracy reference. Plain
Monte Carlo converges like n^-0.5. Latin hypercube lowers the constant. The
Sobol methods improve the rate for this smooth QoI.

Run from the repository root:

    julia --project=benchmark -t auto benchmark/convergence.jl

The MATPOWER case downloads once into benchmark/data/. Results are printed as a
table and serialized to benchmark/convergence.jls for plotting.
"""

using ProbabilisticPowerFlow
using Distributions: LogNormal
using Random: Xoshiro
using Statistics: mean, std
using Serialization: serialize
using Downloads: download
using Base.Threads: @threads
import QuasiMonteCarlo as QMC
using Sobol: Sobol
import PowerModels as PM
PM.silence()

const CASE_URL = "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/pglib_opf_case1354_pegase.m"
const CASE_PATH = joinpath(@__DIR__, "data", "pglib_opf_case1354_pegase.m")

if !isfile(CASE_PATH)
    mkpath(dirname(CASE_PATH))
    @info "downloading case1354 PEGASE" CASE_URL
    download(CASE_URL, CASE_PATH)
end

data = PM.parse_file(CASE_PATH)
backend = PowerModelsBackend(data)

# correlated relative load uncertainty on the 200 largest PQ loads
bustype = Dict(bus["index"] => bus["bus_type"] for bus in values(data["bus"]))
eligible = [
    (id, l) for (id, l) in data["load"] if
    l["status"] == 1 && bustype[l["load_bus"]] == 1 && l["pd"] > 0
]
sort!(eligible; by = p -> -p[2]["pd"])
eligible = eligible[1:min(200, end)]

vars = GermVariable[]
assigns = Assignment[]
dist = LogNormal(0.0, 0.05)
for (id, l) in eligible
    push!(vars, GermVariable("l$id", dist))
    push!(
        assigns,
        Assignment("l$id", ComponentRef(:load, id, :pd), AffineTransform(l["pd"], 0.0)),
    )
end
d = length(vars)
R = fill(0.3, d, d)
for k = 1:d
    R[k, k] = 1.0
end
model = UncertaintyModel(vars, assigns, GaussianCopula(R))

# monitor the PQ bus whose voltage moves most under the uncertainty
pilot = let
    prob = PPFProblem(backend, model, AbstractQoI[])
    pq = [bus["index"] for bus in values(data["bus"]) if bus["bus_type"] == 1]
    qois = AbstractQoI[VoltageMagnitude(b) for b in pq]
    r = solve(
        PPFProblem(backend, model, qois),
        MonteCarlo(n = 128, warmstart = :chain);
        rng = Xoshiro(0),
    )
    stds = [std(qoi_samples(r, q)) for q in qois]
    pq[argmax(stds)]
end
@info "monitoring bus $pilot"

qoi = VoltageMagnitude(pilot)
prob = PPFProblem(backend, model, AbstractQoI[qoi])

scrambled(seed) =
    QMC.SobolSample(R = QMC.OwenScramble(base = 2, pad = 32, rng = Xoshiro(seed)))
methods(n, seed) = [
    ("MonteCarlo", MonteCarlo(; n, warmstart = :chain), Xoshiro(seed)),
    ("LatinHypercube", LatinHypercube(; n, warmstart = :chain), Xoshiro(seed)),
    ("SobolSampling", SobolSampling(; n, warmstart = :chain), Xoshiro(seed)),
    ("QMC_OwenSobol", QMCSampling(scrambled(seed); n, warmstart = :chain), Xoshiro(seed)),
]

const NS = [32, 64, 128, 256, 512, 1024]
const NREP = 8

# high-accuracy reference from an Owen-scrambled Sobol run
@info "reference run"
truth = mean(solve(prob, QMCSampling(scrambled(999); n = 8192, warmstart = :chain)), qoi)
@info "reference mean vm at bus $pilot: $truth"

jobs = [(name, n, rep) for n in NS for rep = 1:NREP for (name, _, _) in methods(2, 1)]
results = Dict{Tuple{String,Int,Int},Float64}()
lk = ReentrantLock()
elapsed = @elapsed @threads for (name, n, rep) in jobs
    entry = only(filter(m -> m[1] == name, methods(n, 10_000 + 97 * rep)))
    est = mean(solve(prob, entry[2]; rng = entry[3]), qoi)
    lock(lk) do
        results[(name, n, rep)] = est
    end
end
@info "$(length(jobs)) runs in $(round(elapsed, digits = 1)) s"

rmse = Dict(
    (name, n) => sqrt(mean([(results[(name, n, r)] - truth)^2 for r = 1:NREP])) for
    (name, n, _) in jobs
)

println("\nRMSE of the mean vm estimate at bus $pilot, $(NREP) replicates:")
println(rpad("n", 8), join(rpad.(first.(methods(2, 1)), 16)))
for n in NS
    print(rpad(n, 8))
    for (name, _, _) in methods(2, 1)
        print(rpad(round(rmse[(name, n)], sigdigits = 3), 16))
    end
    println()
end

serialize(joinpath(@__DIR__, "convergence.jls"), (; NS, NREP, rmse, truth, pilot))
println("\nserialized to benchmark/convergence.jls")
