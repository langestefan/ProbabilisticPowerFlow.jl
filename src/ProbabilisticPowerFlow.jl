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

# solve and solve! are CommonSolve's functions, not ours. CommonSolve is the tiny
# interface package the SciML ecosystem shares, so extending it means loading this
# package next to NonlinearSolve.jl gives one solve rather than a name clash.
import CommonSolve: solve, solve!

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
include("nataf.jl")
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
export AbstractDependence, IndependentCopula, GaussianCopula
export spearman_to_gaussian, pearson_to_gaussian
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
    PowerModelsBackend(data::AbstractDict; solver = :nlsolve, tol = 1e-8,
                       maxiter = 100)

AC power flow backend on a PowerModels.jl network data dictionary, for example the
result of `PowerModels.parse_file`. Newton iterations on a sparse admittance
matrix, with a pluggable solve loop:

  - `solver = :nlsolve`, the default, runs PowerModels' bundled NLsolve path.
  - `solver = NewtonRaphson()` or any other NonlinearSolve.jl algorithm, available
    after `using NonlinearSolve`, runs the same equations through a nonlinear
    cache that is built once per state and reused, so repeated solves allocate
    almost nothing. Concurrent sampling scales best on this path.

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

# Internal seam for the PowerModels backend's solver choice. The PowerModels
# extension implements the bundled NLsolve path for solver == :nlsolve, and the
# NonlinearSolve extension adds a method for NonlinearSolve algorithms with a
# reusable cache. Both live in extensions, so the dispatch functions are owned
# here and extended there.
function run_pf_solver! end
supports_pf_solver(solver) = false

end
