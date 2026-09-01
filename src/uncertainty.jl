"""
    GermVariable(id, dist)

A named random variable of the germ, with marginal distribution `dist`, any
[`UnivariateDistribution`](@extref Distributions :std:doc:`univariate`).

The germ is the set of all germ variables, and is the only source of uncertainty in a
model. Everything else is derived from it, by a transform or by an assignment to a
physical quantity. How germ variables are correlated is described by the copula in
[`UncertaintyModel`](@ref).
"""
struct GermVariable{D<:UnivariateDistribution}
    id::String
    dist::D
end

"""
    Assignment(variable, target[, transform])

Describes how germ variables map to network components.

Germ variables have to be assigned to network components, optionally through a
[`Transform`](@ref). The transform is applied to the germ value before writing it to
the component. Such a transform can for example be used when we specify windspeed as
a germ variable, but use a power curve transform to get the generator's active power
injection. It is allowed to assign the same germ variable to multiple components, which
expresses perfect correlation between those components. An example could be a constant
powerfactor load.
"""
struct Assignment{T<:AbstractTransform}
    variable::String
    target::ComponentRef
    transform::T
end

Assignment(variable::AbstractString, target::ComponentRef) =
    Assignment(String(variable), target, IdentityTransform())

"""
    UncertaintyModel(variables, assignments, dependence)

"""
struct UncertaintyModel{C,V<:AbstractVector{<:GermVariable},A<:AbstractVector{<:Assignment}}
    variables::V
    assignments::A
    dependence::C
    varindex::Vector{Int}   # assignment j reads germ value varindex[j]

    function UncertaintyModel(
        variables::AbstractVector{<:GermVariable},
        assignments::AbstractVector{<:Assignment},
        dependence::C,
    ) where {C} end
end

"""
    germ_dim(m::UncertaintyModel)

Number of germ variables `d`, the dimension of the samplers in `u ∈ (0,1)^d` space.
"""
germ_dim(m::UncertaintyModel) = length(m.variables)

"""
    targets(m::UncertaintyModel)

The `ComponentRef`s of the assignments, in assignment order. This is the `refs`
argument for [`init_state`](@ref), and the order of the physical injection vector.
"""
targets(m::UncertaintyModel) = getfield.(m.assignments, :target)

"""
    to_physical!(x, m::UncertaintyModel, u, germ) -> x

Map one sample of independent uniforms `u ∈ (0,1)^d` to the physical injection
vector `x`.
"""
function to_physical!(
    x::AbstractVector{Float64},
    m::UncertaintyModel,
    u::AbstractVector{<:Real},
    germ::AbstractVector{Float64},
) end

"""
    to_physical(m::UncertaintyModel, u)

Allocating wrapper around [`to_physical!`](@ref).
"""
to_physical(m::UncertaintyModel, u::AbstractVector{<:Real}) = to_physical!(
    Vector{Float64}(undef, length(m.assignments)),
    m,
    u,
    Vector{Float64}(undef, germ_dim(m)),
)
