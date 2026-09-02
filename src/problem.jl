"""
    PPFProblem(backend, model, qois)

A probabilistic power flow problem: a deterministic solver `backend`, an
[`UncertaintyModel`](@ref) on its injections, and the quantities of interest to
estimate.

A problem is the *what*, a method is the *how*. Any
[`AbstractPPFMethod`](@ref) consumes a problem and produces a [`PPFResult`](@ref),
so the same problem can be run by several methods and their estimates compared.
"""
struct PPFProblem{B<:AbstractPFBackend}
    backend::B
    model::UncertaintyModel
    qois::Vector{AbstractQoI}
end

PPFProblem(backend::AbstractPFBackend, model::UncertaintyModel, qois::AbstractVector) =
    PPFProblem(backend, model, collect(AbstractQoI, qois))
