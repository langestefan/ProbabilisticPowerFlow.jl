module PPFQuasiMonteCarloExt

import QuasiMonteCarlo as QMC
import ProbabilisticPowerFlow as PPF

# The sampler owns its point set and randomization, so no rng is threaded
# through. Points are clamped away from 0 and 1 because unrandomized sequences
# can contain the origin and marginal quantiles of unbounded distributions must
# stay finite.
function solve_qmc(prob::PPF.PPFProblem, method::PPF.QMCSampling)
    PPF.check_warmstart(method.warmstart, prob.backend)
    d = PPF.germ_dim(prob.model)
    U = Matrix{Float64}(QMC.sample(method.n, d, method.sampler))
    clamp!(U, eps(), 1 - eps())
    return PPF.solve_u_matrix(prob, method, U)
end

end
