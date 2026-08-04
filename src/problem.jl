"""
    PPFProblem(backend, model, qois)

A probabilistic power flow problem: a deterministic solver backend, an uncertainty
model on its injections, and the quantities of interest to estimate. Methods
([`AbstractPPFMethod`](@ref)) consume a problem and produce a [`PPFResult`](@ref).
"""
struct PPFProblem{B<:AbstractBackend}
    backend::B
    model::UncertaintyModel
    qois::Vector{AbstractQoI}
end

PPFProblem(backend::AbstractBackend, model::UncertaintyModel, qois::AbstractVector) =
    PPFProblem(backend, model, collect(AbstractQoI, qois))
