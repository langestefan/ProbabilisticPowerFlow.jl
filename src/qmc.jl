"""
    QMCSampling(sampler; n = 1000, warmstart = :off, keep_inputs = false)

Quasi-Monte Carlo sampling with any point-set generator from QuasiMonteCarlo.jl,
for example `SobolSample`, `HaltonSample`, or `LatticeRuleSample`. Randomization
is owned by the sampler. Pass a QuasiMonteCarlo randomization method inside the
sampler and give it an rng to make the point set reproducible, for example

    QMCSampling(SobolSample(R = OwenScramble(base = 2, pad = 32, rng = rng)))

The `rng` keyword of [`solve`](@ref) does not affect the points. Owen scrambling
in base 2 requires a power-of-2 sample count.

A sampler also passes directly into `solve`, with the configuration as keywords:

    solve(prob, SobolSample(); n = 1024, warmstart = :chain)

This is equivalent to wrapping it in `QMCSampling` and the result records the
full configuration either way.

The implementation lives in the `PPFQuasiMonteCarloExt` package extension and
needs `using QuasiMonteCarlo`. [`SobolSampling`](@ref) stays the light default for
a shifted Sobol sequence without the extra dependencies.

`warmstart` and `keep_inputs` behave as in [`MonteCarlo`](@ref).
"""
struct QMCSampling{S} <: AbstractPPFMethod
    sampler::S
    n::Int
    warmstart::Symbol
    keep_inputs::Bool
end

QMCSampling(
    sampler;
    n::Integer = 1000,
    warmstart::Symbol = :off,
    keep_inputs::Bool = false,
) = QMCSampling(sampler, Int(n), warmstart, keep_inputs)

function solve(
    prob::PPFProblem,
    method::QMCSampling;
    rng::AbstractRNG = Random.default_rng(),
    ntasks::Integer = 1,
)
    ext = Base.get_extension(@__MODULE__, :PPFQuasiMonteCarloExt)
    ext === nothing && throw(
        ArgumentError(
            "QMCSampling requires the QuasiMonteCarlo package. Run " *
            "`using QuasiMonteCarlo` first.",
        ),
    )
    return ext.solve_qmc(prob, method, ntasks)
end
