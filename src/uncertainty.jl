"""
    GermVariable(id, dist)

A basic random variable of the germ, with a univariate distribution in its natural
space, such as a wind speed in m/s or a relative load level. The germ is the finite
set of basic random variables from which all uncertainty in the model is generated.
Everything downstream is a deterministic function of it.
"""
struct GermVariable{D<:UnivariateDistribution}
    id::String
    dist::D
end

"""
    Assignment(variable, target[, transform])

How a germ variable lands on a network component: the germ value of `variable` (a
`GermVariable` id) is passed through `transform` and written to `target`. Several
assignments may reference one germ variable — sharing a variable expresses perfectly
coupled quantities, such as P and Q of a constant-power-factor load, without
degenerate correlation entries.
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

The full uncertainty description: germ variables, assignments onto the network, and
a copula on the germ. Validates on construction:

  - every assignment references a declared germ variable,
  - every germ variable is referenced by at least one assignment,
  - the dependence dimension, when the copula declares one, equals the number of
    germ variables.

Component existence in the network is the backend's half of the validation, enforced
by [`init_state`](@ref).
"""
struct UncertaintyModel{
    C<:AbstractDependence,
    V<:AbstractVector{<:GermVariable},
    A<:AbstractVector{<:Assignment},
}
    variables::V
    assignments::A
    dependence::C
    varindex::Vector{Int}   # assignment j reads germ value varindex[j]

    function UncertaintyModel(
        variables::AbstractVector{<:GermVariable},
        assignments::AbstractVector{<:Assignment},
        dependence::C,
    ) where {C<:AbstractDependence}
        ids = [v.id for v in variables]
        length(unique(ids)) == length(ids) ||
            throw(ArgumentError("duplicate germ variable ids: $(ids)"))
        index = Dict(id => k for (k, id) in enumerate(ids))
        varindex = Vector{Int}(undef, length(assignments))
        for (j, a) in enumerate(assignments)
            k = get(index, a.variable, 0)
            k == 0 && throw(
                ArgumentError(
                    "assignment $(j) references undeclared germ variable " *
                    "$(repr(a.variable))",
                ),
            )
            varindex[j] = k
        end
        referenced = falses(length(variables))
        referenced[varindex] .= true
        if !all(referenced)
            unused = ids[.!referenced]
            throw(ArgumentError("germ variables referenced by no assignment: $(unused)"))
        end
        d = dependence_dim(dependence)
        if d !== nothing && d != length(variables)
            throw(
                ArgumentError(
                    "dependence dimension $(d) does not match the number of germ " *
                    "variables $(length(variables))",
                ),
            )
        end
        # plain collect keeps a concrete element type when the variables or
        # assignments are homogeneous, which removes dynamic dispatch from the
        # to_physical! hot loop
        vars = collect(variables)
        assigns = collect(assignments)
        return new{C,typeof(vars),typeof(assigns)}(vars, assigns, dependence, varindex)
    end
end

"""
    germ_dim(m::UncertaintyModel)

Number of germ variables `d` — the dimension of the samplers' `u ∈ (0,1)^d` space.
The name avoids a clash with `Distributions.dim`.
"""
germ_dim(m::UncertaintyModel) = length(m.variables)

"""
    targets(m::UncertaintyModel)

The `ComponentRef`s of the assignments, in assignment order. This is the `refs`
argument for [`init_state`](@ref), and the order of the physical injection vector.
"""
targets(m::UncertaintyModel) = [a.target for a in m.assignments]

"""
    to_physical!(x, m::UncertaintyModel, u, germ) -> x

Map one sample of independent uniforms `u ∈ (0,1)^d` to the physical injection
vector `x`. This is the pipeline every sampling method uses. It consists of three steps:

 1. [`to_dependent!`](@ref) applies the copula: `u` becomes dependent uniforms
    whose ranks carry the correlation structure.
 2. Each variable's marginal quantile function turns its dependent uniform into
    a value with that variable's distribution; the result is the germ.
 3. Each assignment's transform maps its germ value to a physical injection
    `x[j]`, in assignment order.

Only `u` is random; everything here is deterministic, so equal `u` gives equal `x`.
`germ` is a length-`germ_dim(m)` work buffer, `x` a length-`length(m.assignments)`
output buffer.
"""
function to_physical!(
    x::AbstractVector{Float64},
    m::UncertaintyModel,
    u::AbstractVector{<:Real},
    germ::AbstractVector{Float64},
)
    length(u) == germ_dim(m) || throw(
        DimensionMismatch(
            "u has length $(length(u)), expected germ dimension $(germ_dim(m))",
        ),
    )
    to_dependent!(germ, m.dependence, u)
    @inbounds for k = 1:germ_dim(m)
        germ[k] = quantile(m.variables[k].dist, germ[k])
    end
    @inbounds for j in eachindex(m.assignments)
        x[j] = m.assignments[j].transform(germ[m.varindex[j]])
    end
    return x
end

"""
    to_physical(m::UncertaintyModel, u)

Allocating convenience wrapper around [`to_physical!`](@ref).
"""
to_physical(m::UncertaintyModel, u::AbstractVector{<:Real}) = to_physical!(
    Vector{Float64}(undef, length(m.assignments)),
    m,
    u,
    Vector{Float64}(undef, germ_dim(m)),
)
