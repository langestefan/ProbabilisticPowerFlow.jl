"""
    ProbabilisticPowerFlow

A method-agnostic framework for probabilistic power flow: one problem definition,
swappable sampling methods, swappable deterministic solver backends. Every method
uses the same pipeline

    u ~ U(0,1)^d → germ (copula + marginal quantiles)
                 → physical injections (transforms)
                 → power flow solution (backend)

Only `u` is random. Everything after it is deterministic, so equal `u` gives equal
injections, and a new sampling method is a new rule for producing `u` and nothing
else.
"""
module ProbabilisticPowerFlow

# solve and solve! are CommonSolve's functions, not ours. CommonSolve is the tiny
# interface package the SciML ecosystem shares, so extending it means loading this
# package next to NonlinearSolve.jl gives one solve rather than a name clash.
import CommonSolve: solve, solve!

using Distributions: UnivariateDistribution, quantile
using Copulas: IndependentCopula, inverse_rosenblatt, SklarDist
using EnumX: @enumx
using Random: Random, AbstractRNG, rand!
using Statistics: Statistics, mean, quantile, std

include("backend_interface.jl")
include("qoi.jl")
include("transform.jl")
include("uncertainty.jl")
include("problem.jl")
include("methods.jl")
include("result.jl")
include("sample_loop.jl")
include("monte_carlo.jl")
include("show.jl")

# Exported symbols needed to implement the backend interface
export ComponentRef, ComponentField, ComponentKind, SolveInfo, kind
export AbstractPFBackend, init_state, set_injections!, solve!, extract
export supports_warmstart, linearize

# Quantities of interest
export AbstractQoI, VoltageMagnitude, VoltageAngle
export BranchActivePower, BranchReactivePower, ViolationEvent

# Uncertainty model
export AbstractTransform, IdentityTransform, AffineTransform
export GermVariable, Assignment, UncertaintyModel, germ_dim, germ_dist, targets
export to_physical, to_physical!

# Problem, methods, results
export PPFProblem, AbstractPPFMethod, MonteCarlo, solve
export PPFResult, FailedSample, n_converged, failure_rate
export qoi_samples, violation_probability

"""
    PowerModelsBackend(data::AbstractDict; solver = NativeNewton(), tol = 1e-8,
                       maxiter = 50)

AC power flow backend on a PowerModels.jl network data dictionary, for example the
result of `PowerModels.parse_file`.

The constructor lives in the `PPFPowerModelsExt` package extension, so it is only
available after `using PowerModels`. Without PowerModels loaded, calling it throws a
`MethodError`.

`solver` is any algorithm PowerModels can solve a `PowerFlowSystem` with: its
built-in `NativeNewton()`, or a NonlinearSolve.jl algorithm such as
`NewtonRaphson()` once `using NonlinearSolve` has loaded PowerModels' own SciML
extension. `tol` is the tolerance on the infinity-norm of the power mismatch and
`maxiter` the iteration limit; both are passed to the solver.

Injections may target `Pd`, `Qd` and `Pg`. Setpoints (`Vg`, `Vm`) and `Qg` are not
injections in this formulation and are rejected by [`init_state`](@ref).
"""
function PowerModelsBackend end

export PowerModelsBackend

end
