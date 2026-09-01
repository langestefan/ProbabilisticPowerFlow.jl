module ProbabilisticPowerFlow

using EnumX: @enumx
using Distributions: UnivariateDistribution

import CommonSolve: solve!

include("backend_interface.jl")
include("qoi.jl")
include("transform.jl")
include("uncertainty.jl")

# Exported symbols needed to implement the backend interface
export ComponentRef, ComponentField, ComponentKind, SolveInfo, kind
export AbstractPFBackend, init_state, set_injections!, solve!, extract
export supports_warmstart, linearize

# Quantities of interest
export AbstractQoI, VoltageMagnitude, VoltageAngle
export BranchActivePower, BranchReactivePower, ViolationEvent

# Uncertainty model
# export AbstractTransform, IdentityTransform, AffineTransform
# export AbstractDependence, IndependentCopula, GaussianCopula
# export spearman_to_gaussian, pearson_to_gaussian
export GermVariable, Assignment, UncertaintyModel, germ_dim, targets
export to_physical, to_physical!

end
