"""
    AbstractDependence

Dependence structure (copula) on the germ variables. The copula always lives on the
germ, never on transformed outputs.

Subtypes implement `to_dependent!(v, dep, u)` mapping independent uniforms `u ∈ (0,1)^d`
to dependent uniforms `v`, and `dependence_dim(dep)`, which returns the germ dimension
`d`, or `nothing` when the structure is dimension-agnostic.

Future subtypes slot in here without touching any sampler: a t-copula, a
Copulas.jl wrapper, and a Nataf-corrected Gaussian copula for Pearson targets.
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

`:pearson` is deliberately rejected: matching Pearson correlation in physical space
requires an iterative Nataf correction per marginal family, which is not implemented
yet, and silently reinterpreting the matrix is how benchmark results go wrong.

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
                "correlation = :pearson requires the Nataf correction, which is not " *
                "implemented. Use :spearman or supply the copula parameter directly " *
                "via :gaussian",
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
