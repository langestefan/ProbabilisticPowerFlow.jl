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

Germ variables have to be assigned to network components, optionally through an
[`AbstractTransform`](@ref). The transform is applied to the germ value before writing it to
the component. Such a transform can for example be used when we specify windspeed as
a germ variable, but use a power curve transform to get the generator's active power
injection.

It is allowed to assign the same germ variable to multiple components, which expresses
perfect correlation between those components. An example could be a constant powerfactor
load.
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
    UncertaintyModel(variables, assignments)

The full description of the uncertainty in a model: the germ variables, their mapping
onto network components through `assignments`, and the dependence structure between the
germ variables.

The dependence structure is a [copula](@extref Copulas :std:doc:`manual/intro`), a
multivariate distribution with uniform marginals. Its dimension must equal the number of
germ variables, because the two together are one joint distribution. The two-argument
form defaults to an `IndependentCopula` of the right dimension.

Internally we cache `varindex`, a static mapping from assignment order to germ variable
order, so [`to_physical!`](@ref) reads an integer index instead of resolving a name once
per assignment per sample.

Validated on construction: germ variable ids are unique, every assignment references a
declared germ variable, every germ variable is referenced by at least one assignment, and
the dependence dimension matches. Component existence in the network is the backend's
half, enforced by [`init_state`](@ref).
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
    ) where {C}
        ids = getfield.(variables, :id)
        if length(unique(ids)) != length(ids)
            throw(ArgumentError("duplicate germ variable ids: $(ids)"))
        end

        # build the mapping from assignment order to germ variable order
        index = Dict(id => k for (k, id) in enumerate(ids))
        varindex = map(enumerate(assignments)) do (j, a)
            k = get(index, a.variable, 0)
            if k == 0
                throw(
                    ArgumentError(
                        "assignment $(j) references undeclared germ variable " *
                        "$(repr(a.variable))",
                    ),
                )
            end
            return k
        end

        # check that we have no floating variables (variable without an assignment)
        referenced = falses(length(variables))
        referenced[varindex] .= true
        if !all(referenced)
            throw(
                ArgumentError(
                    "germ variables referenced by no assignment: $(ids[.!referenced])",
                ),
            )
        end

        d = length(dependence)
        if d != length(variables)
            throw(
                ArgumentError(
                    "dependence dimension $(d) does not match the number of germ " *
                    "variables $(length(variables))",
                ),
            )
        end

        # collect keeps a concrete element type when the inputs are homogeneous
        vars = collect(variables)
        assigns = collect(assignments)
        return new{C,typeof(vars),typeof(assigns)}(vars, assigns, dependence, varindex)
    end
end

UncertaintyModel(
    variables::AbstractVector{<:GermVariable},
    assignments::AbstractVector{<:Assignment},
) = UncertaintyModel(variables, assignments, IndependentCopula(length(variables)))

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
)
    # loop over dimension validations
    for (name, v, n) in (
        ("u", u, germ_dim(m)),
        ("germ", germ, germ_dim(m)),
        ("x", x, length(m.assignments)),
    )
        if length(v) != n
            throw(DimensionMismatch("$(name) has length $(length(v)), expected $(n)"))
        end
    end

    # from independent uniforms to correlated uniforms
    germ .= inverse_rosenblatt(m.dependence, u)

    # inverse transform sampling: F⁻¹(u) has distribution F when u ~ U(0,1)
    # (quantile is the inverse CDF = F⁻¹)
    @inbounds for k in eachindex(germ)
        germ[k] = quantile(m.variables[k].dist, clamp(germ[k], eps(), 1 - eps()))
    end

    # map to physical injections using the assignment transforms
    # if the germ models the physical injection directly this is equal to identity
    @inbounds for j in eachindex(m.assignments)
        x[j] = m.assignments[j].transform(germ[m.varindex[j]])
    end

    return x
end

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

"""
    germ_dist(m::UncertaintyModel)

The joint distribution of the germ variables, as a
[`SklarDist`](@extref Copulas Copulas.SklarDist).
"""
germ_dist(m::UncertaintyModel) = SklarDist(m.dependence, Tuple(v.dist for v in m.variables))
