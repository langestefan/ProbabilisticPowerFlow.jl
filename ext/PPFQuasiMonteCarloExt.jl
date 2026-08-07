module PPFQuasiMonteCarloExt

import QuasiMonteCarlo as QMC
import ProbabilisticPowerFlow as PPF
import Random

# A QuasiMonteCarlo sampler passes directly into solve, the way foreign types
# are meant to work with extensions. The kwargs carry the run configuration the
# sampler itself does not hold. Internally this builds the QMCSampling record,
# so PPFResult.method still documents the full configuration of the run.
function PPF.solve(
    prob::PPF.PPFProblem,
    sampler::QMC.SamplingAlgorithm;
    n::Integer = 1000,
    warmstart::Symbol = :off,
    keep_inputs::Bool = false,
    rng::Random.AbstractRNG = Random.default_rng(),
    ntasks::Integer = 1,
)
    return PPF.solve(prob, PPF.QMCSampling(sampler; n, warmstart, keep_inputs); rng, ntasks)
end

# The sampler owns its point set and randomization, so no rng is threaded
# through. Points are clamped away from 0 and 1 because unrandomized sequences
# can contain the origin and marginal quantiles of unbounded distributions must
# stay finite.
function solve_qmc(prob::PPF.PPFProblem, method::PPF.QMCSampling, ntasks)
    PPF.check_warmstart(method.warmstart, prob.backend)
    d = PPF.germ_dim(prob.model)
    U = Matrix{Float64}(QMC.sample(method.n, d, method.sampler))
    clamp!(U, eps(), 1 - eps())
    return PPF.solve_u_matrix(prob, method, U; ntasks)
end

end
