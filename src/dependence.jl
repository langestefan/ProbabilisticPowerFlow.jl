"""
    AbstractDependence

Dependence structure (copula) on the germ variables. The copula always lives on the
germ, never on transformed outputs.

The dependence interface is duck typed on two functions: `to_dependent!(v, dep, u)`
maps independent uniforms `u ∈ (0,1)^d` to dependent uniforms `v`, and
`dependence_dim(dep)` returns the germ dimension `d`, or `nothing` when the
structure is dimension-agnostic. The built-in structures subtype this abstract
type, but [`UncertaintyModel`](@ref) accepts any type whose package implements the
two functions.

With `using Copulas`, every copula from Copulas.jl works directly as the
dependence, for example `Copulas.ClaytonCopula(d, 2.0)`. The map is the
deterministic inverse Rosenblatt transform, so all sampling methods compose,
including the quasi-Monte Carlo ones, and failed samples replay exactly. The
copula parameter means whatever Copulas.jl defines for the family. It is not
Spearman-mapped like the built-in [`GaussianCopula`](@ref). For plain Gaussian
dependence prefer the built-in, which caches its Cholesky factor. Copulas.jl also
exports names `GaussianCopula` and `IndependentCopula`, so qualify those names
when both packages are loaded.
"""
abstract type AbstractDependence end

"""
    to_dependent!(v, dep, u) -> v

Apply the dependence structure `dep`: map independent uniforms `u ∈ (0,1)^d` to
dependent uniforms `v` whose ranks carry the correlation of `dep`. Each entry of `v`
stays uniform on `(0,1)`. This is step 1 of [`to_physical!`](@ref).
"""
function to_dependent! end

"""
    dependence_dim(dep)

The germ dimension a dependence structure is defined for, or `nothing` when the
structure is dimension-agnostic. [`UncertaintyModel`](@ref) checks it against the
number of germ variables at construction.
"""
function dependence_dim end

"""
    IndependentCopula()

Independent germ variables; `to_dependent!` is the identity.
"""
struct IndependentCopula <: AbstractDependence end

dependence_dim(::IndependentCopula) = nothing

to_dependent!(v::AbstractVector{Float64}, ::IndependentCopula, u::AbstractVector{<:Real}) =
    copyto!(v, u)

"""
    spearman_to_gaussian(rho_s)

Closed-form map from a Spearman rank correlation to the Gaussian-copula parameter:
`rho_g = 2 sin(pi * rho_s / 6)`.
"""
spearman_to_gaussian(rho_s::Real) = 2 * sin(pi * rho_s / 6)

"""
    GaussianCopula(R; correlation = :spearman)

Gaussian copula on the germ. `R` is a symmetric correlation matrix interpreted
according to `correlation`:

  - `:spearman` (default): rank correlations, mapped elementwise to the copula
    parameter via [`spearman_to_gaussian`](@ref). Rank correlation is the package
    default: it is unambiguous and invariant under the monotone parts of the
    transforms.
  - `:gaussian`: `R` is used directly as the copula parameter.

A Pearson target needs the marginals, because the copula parameter that induces it
depends on them. Pass them as a second argument, which selects the Nataf correction:
`GaussianCopula(R, variables; correlation = :pearson)`. This constructor never
reinterprets a matrix silently, since that is how benchmark results go wrong.

Positive semi-definiteness is validated *after* the mapping. On failure the error
names the offending eigenvalue. Projection to the nearest PSD matrix is never done
silently.
"""
struct GaussianCopula <: AbstractDependence
    L::LowerTriangular{Float64,Matrix{Float64}}
end

function GaussianCopula(R::AbstractMatrix{<:Real}; correlation::Symbol = :spearman)
    size(R, 1) == size(R, 2) ||
        throw(ArgumentError("correlation matrix must be square, got size $(size(R))"))
    issymmetric(R) || throw(ArgumentError("correlation matrix must be symmetric"))
    all(isapprox.(diag(R), 1.0; atol = 1e-12)) ||
        throw(ArgumentError("correlation matrix must have a unit diagonal"))
    Rg = if correlation == :spearman
        spearman_to_gaussian.(R)
    elseif correlation == :gaussian
        Matrix{Float64}(R)
    elseif correlation == :pearson
        throw(
            ArgumentError(
                "correlation = :pearson needs the marginals, because the Nataf " *
                "correction from a Pearson target to the copula parameter depends " *
                "on them. Pass them as a second argument: " *
                "GaussianCopula(R, variables; correlation = :pearson)",
            ),
        )
    else
        throw(ArgumentError("unknown correlation type $(repr(correlation))"))
    end
    # The elementwise Spearman map preserves the unit diagonal but not PSD-ness in
    # general, so validate after mapping.
    for i in axes(Rg, 1)
        Rg[i, i] = 1.0
    end
    lambda_min = eigmin(Symmetric(Rg))
    if lambda_min < -1e-10
        throw(
            ArgumentError(
                "correlation matrix is not positive semi-definite after the " *
                "$(correlation) → Gaussian-copula mapping: smallest eigenvalue is " *
                "$(lambda_min). Fix the input correlations. PSD projection is only " *
                "available as an explicit opt-in, never silently.",
            ),
        )
    end
    chol = cholesky(Symmetric(Rg + 1e-12 * I))
    return GaussianCopula(LowerTriangular(chol.L))
end

dependence_dim(c::GaussianCopula) = size(c.L, 1)

function to_dependent!(
    v::AbstractVector{Float64},
    c::GaussianCopula,
    u::AbstractVector{<:Real},
)
    stdnormal = Normal()
    @inbounds for k in eachindex(v)
        v[k] = quantile(stdnormal, u[k])
    end
    z = c.L * v
    @inbounds for k in eachindex(v)
        v[k] = cdf(stdnormal, z[k])
    end
    return v
end
