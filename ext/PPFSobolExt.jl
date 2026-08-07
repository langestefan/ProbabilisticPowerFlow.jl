module PPFSobolExt

import Sobol
import ProbabilisticPowerFlow as PPF

# Sobol points with a random Cranley-Patterson shift: u = (s + shift) mod 1 per
# dimension. The shift preserves the low-discrepancy structure, makes the
# estimate unbiased, and ties it to the rng seed. Clamped away from 0 and 1 so
# marginal quantiles of unbounded distributions stay finite.
function solve_sobol(prob::PPF.PPFProblem, method::PPF.SobolSampling, rng, ntasks)
    PPF.check_warmstart(method.warmstart, prob.backend)
    d = PPF.germ_dim(prob.model)
    n = method.n
    seq = Sobol.SobolSeq(d)
    shift = rand(rng, d)
    U = Matrix{Float64}(undef, d, n)
    p = Vector{Float64}(undef, d)
    for i = 1:n
        Sobol.next!(seq, p)
        for k = 1:d
            U[k, i] = mod(p[k] + shift[k], 1.0)
        end
    end
    clamp!(U, eps(), 1 - eps())
    return PPF.solve_u_matrix(prob, method, U; ntasks)
end

end
