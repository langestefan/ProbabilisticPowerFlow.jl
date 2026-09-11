"""
    PPFProblem(backend, model, qois)

The definition of a probabilistic power flow problem.

A `PPFProblem` contains a deterministic power flow backend, an uncertainty
model on its injections, and the quantities of interest we are looking to estimate.

A method ([`AbstractPPFMethod`](@ref)) takes a problem and returns a [`PPFResult`](@ref).
"""
struct PPFProblem{B <: AbstractPFBackend, M <: UncertaintyModel}
    backend::B
    model::M
    qois::Vector{AbstractQoI}
end

PPFProblem(backend::AbstractPFBackend, model::UncertaintyModel, qois::AbstractVector) =
    PPFProblem(backend, model, collect(AbstractQoI, qois))
