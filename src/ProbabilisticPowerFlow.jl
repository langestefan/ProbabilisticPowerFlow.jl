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
using Random: Random, AbstractRNG, rand!, randperm
using Statistics: Statistics, mean, quantile, std

include("backend_interface.jl")
include("qoi.jl")
include("transforms.jl")
include("dependence.jl")
include("uncertainty.jl")
include("problem.jl")
include("methods.jl")
include("result.jl")
include("sample_loop.jl")
include("monte_carlo.jl")
include("latin_hypercube.jl")
include("sobol.jl")
include("qmc.jl")
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
export PPFProblem,
    AbstractPPFMethod, MonteCarlo, LatinHypercube, SobolSampling, QMCSampling, solve
export PPFResult, FailedSample, n_converged, failure_rate
export qoi_samples, violation_probability

# Reference backend
export NetworkData, Branch, build_ybus, ReferenceBackend, case5

# Extension backend stubs. The constructors live in package extensions, which
# cannot export new names, so the empty functions are declared here.

"""
    PowerModelsBackend(data::AbstractDict; tol = 1e-8, maxiter = 100)

AC power flow backend on a PowerModels.jl network data dictionary, for example the
result of `PowerModels.parse_file`. The solver is PowerModels' native Newton method
on a sparse admittance matrix.

The constructor lives in the `PPFPowerModelsExt` package extension, so it is only
available after `using PowerModels`. Without PowerModels loaded, calling this
function throws a `MethodError`.

`tol` is the tolerance on the infinity-norm of the power mismatch. `maxiter` is the
solver iteration limit.

Networks with several reference buses are supported. Each reference bus holds its
voltage magnitude setpoint with the angle fixed at zero, and its active and reactive
injection are outcomes of the solve. This matches how pandapower treats several
external grid connections, which is what benchmark datasets such as SimBench assume.
"""
function PowerModelsBackend end

export PowerModelsBackend

end
