"""
    ProbabilisticPowerFlow

A method-agnostic framework for probabilistic power flow: one problem definition,
swappable computation methods, swappable deterministic solver backends. Every
sampling method uses the same pipeline

    u ~ U(0,1)^d → germ (copula + marginal quantiles)
                 → physical injections (transforms)
                 → power flow solution (backend)

See `archive/ppf_design.md` in the repository for the design document.
"""
module ProbabilisticPowerFlow

using Distributions: Distributions, Normal, UnivariateDistribution, cdf, quantile
using LinearAlgebra:
    LinearAlgebra,
    I,
    LowerTriangular,
    Symmetric,
    cholesky,
    diag,
    eigmin,
    issuccess,
    issymmetric,
    lu,
    norm
using Random: Random, AbstractRNG, rand!
using Statistics: Statistics, mean, quantile, std

include("backend_interface.jl")
include("qoi.jl")
include("transforms.jl")
include("dependence.jl")
include("uncertainty.jl")
include("problem.jl")
include("methods.jl")
include("result.jl")
include("monte_carlo.jl")
include("reference_backend.jl")
include("case5.jl")

# Backend contract
export AbstractBackend, ComponentRef, SolveInfo
export init_state, set_injections!, solve!, extract, supports_warmstart, linearize

# Quantities of interest
export AbstractQoI, VoltageMagnitude, VoltageAngle, BranchActivePower, ViolationEvent

# Uncertainty model
export AbstractTransform, IdentityTransform, AffineTransform
export AbstractDependence, IndependentCopula, GaussianCopula, spearman_to_gaussian
export GermVariable, Assignment, UncertaintyModel, germ_dim, targets
export to_physical, to_physical!

# Problem, methods, results
export PPFProblem, AbstractPPFMethod, MonteCarlo, solve
export PPFResult, FailedSample, n_converged, failure_rate
export qoi_samples, violation_probability

# Reference backend
export NetworkData, Branch, build_ybus, ReferenceBackend, case5

end
