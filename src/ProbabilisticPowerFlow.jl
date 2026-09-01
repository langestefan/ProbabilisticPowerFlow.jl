module ProbabilisticPowerFlow

using EnumX: @enumx

import CommonSolve: solve!

include("backend_interface.jl")
include("qoi.jl")

# Exported symbols needed to implement the backend interface
export ComponentRef, ComponentField, ComponentKind, SolveInfo, kind
export AbstractPFBackend, init_state, set_injections!, solve!, extract
export supports_warmstart, linearize

# Quantities of interest
export AbstractQoI, VoltageMagnitude, VoltageAngle
export BranchActivePower, BranchReactivePower, ViolationEvent

end
