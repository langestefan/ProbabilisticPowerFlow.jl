module PPFQuasiMonteCarloExt

using CommonSolve: CommonSolve
using ProbabilisticPowerFlow:
    ProbabilisticPowerFlow, PPFProblem, QuasiMC, germ_dim, solve_samples
using QuasiMonteCarlo: SamplingAlgorithm, sample

ProbabilisticPowerFlow.QuasiMC(sampler::SamplingAlgorithm; n::Integer = 1024) =
    QuasiMC(sampler, n)

function CommonSolve.solve(prob::PPFProblem, method::QuasiMC)
    d = germ_dim(prob.model)
    U = Matrix{Float64}(sample(method.n, d, method.sampler))
    clamp!(U, eps(), 1 - eps())
    return solve_samples(prob, method, U)
end

end
