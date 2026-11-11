module PPFImportanceSamplersExt

using ImportanceSamplers: ImportanceSamplers
using ProbabilisticPowerFlow: ProbabilisticPowerFlow, AdaptiveDesignPointSampling, ViolationEvent

ProbabilisticPowerFlow.AdaptiveDesignPointSampling(
    event::ViolationEvent;
    n::Integer = 1000,
    rounds::Integer = 5,
    round_size::Integer = 200,
    defensive::Real = 0.1,
    shift = nothing,
    proposal = nothing,
    fd_step::Real = 0.1,
    maxiter::Integer = 20,
    tol::Real = 1.0e-3,
) = AdaptiveDesignPointSampling(
    event, n, rounds, round_size, defensive, shift, proposal, fd_step, maxiter, tol,
)

end
