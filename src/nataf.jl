# The Nataf correction. A Gaussian copula with parameter rho_0 on the germ induces
# a Pearson correlation rho between the germ variables that depends on their
# marginals, and only equals rho_0 when both marginals are Gaussian. Matching a
# Pearson target therefore means inverting
#
#     rho(rho_0) = E[ g_i(Z_i) g_j(Z_j) ],   (Z_i, Z_j) ~ N(0, [1 rho_0; rho_0 1])
#
# where g_k(z) = (F_k^{-1}(Phi(z)) - mu_k) / sigma_k is the standardized marginal
# pulled back to standard normal space. The expectation is a product Gauss-Hermite
# rule, and rho is increasing in rho_0, so the inversion is a bisection.

"""
    gauss_hermite(n) -> (nodes, weights)

`n`-point Gauss-Hermite quadrature for the standard normal weight, from the
Golub-Welsch eigenvalue problem on the probabilists' Hermite recurrence. The
weights sum to one, so `sum(w .* f.(z))` approximates `E[f(Z)]` for `Z ~ N(0,1)`.
"""
function gauss_hermite(n::Integer)
    n >= 2 || throw(ArgumentError("need at least 2 quadrature nodes, got $(n)"))
    jacobi = LinearAlgebra.SymTridiagonal(zeros(n), sqrt.(1.0:(n-1)))
    factorization = LinearAlgebra.eigen(jacobi)
    return factorization.values, vec(factorization.vectors[1, :]) .^ 2
end

# The standardized marginal in standard normal space. The clamp keeps the tail
# nodes off the exact 0 and 1 that would send a quantile to infinity, which starts
# to matter beyond about 60 nodes.
function standardized_marginal(d::UnivariateDistribution, k::Integer)
    mu = mean(d)
    sd = std(d)
    (isfinite(mu) && isfinite(sd) && sd > 0) || throw(
        ArgumentError(
            "the Nataf correction needs a finite mean and a finite positive standard " *
            "deviation, but marginal $(k) ($(d)) has mean $(mu) and standard " *
            "deviation $(sd)",
        ),
    )
    return z -> (quantile(d, clamp(cdf(Normal(), z), eps(), 1 - eps())) - mu) / sd
end

# rho(rho_0) on the product rule. The conditional form z_j = rho_0 z_a + sqrt(1 -
# rho_0^2) z_b turns the correlated bivariate expectation into two independent
# standard normal integrals, so one set of nodes serves every rho_0.
function induced_correlation(gi_at_nodes, gj, z, w, rho0::Float64)
    scale = sqrt(max(0.0, 1 - rho0^2))
    total = 0.0
    @inbounds for a in eachindex(z)
        inner = 0.0
        for b in eachindex(z)
            inner += w[b] * gj(rho0 * z[a] + scale * z[b])
        end
        total += w[a] * gi_at_nodes[a] * inner
    end
    return total
end

function nataf_pair(rho::Float64, gi_at_nodes, gj, z, w, i::Integer, j::Integer)
    rho == 0 && return 0.0
    lo = induced_correlation(gi_at_nodes, gj, z, w, -1.0)
    hi = induced_correlation(gi_at_nodes, gj, z, w, 1.0)
    if !(lo <= rho <= hi)
        throw(
            ArgumentError(
                "the target Pearson correlation $(rho) for germ pair ($(i), $(j)) is " *
                "outside the range these marginals can attain, which is " *
                "[$(round(lo, digits = 4)), $(round(hi, digits = 4))]. Skewed " *
                "marginals cannot reach every correlation, so the target itself is " *
                "the thing to fix.",
            ),
        )
    end
    a = -1.0
    b = 1.0
    # rho is increasing in rho_0, so bisection cannot lose the root, and 60
    # iterations put it at machine precision on this interval
    for _ = 1:60
        b - a < 1e-12 && break
        mid = (a + b) / 2
        if induced_correlation(gi_at_nodes, gj, z, w, mid) < rho
            a = mid
        else
            b = mid
        end
    end
    return (a + b) / 2
end

"""
    pearson_to_gaussian(rho, di, dj; nodes = 32)
    pearson_to_gaussian(R, marginals; nodes = 32)

Nataf correction: the Gaussian-copula parameter that induces the target Pearson
correlation `rho` between two germ variables with marginals `di` and `dj`, or the
corrected parameter matrix for a full target matrix `R` and its `marginals`. This
is the Pearson counterpart of the closed-form [`spearman_to_gaussian`](@ref).

`marginals` is a vector of `Distributions.jl` distributions or of
[`GermVariable`](@ref)s, in germ variable order. Zero targets map to exactly zero,
and Gaussian marginals reproduce `R` exactly.

The target is on the *germ*. Since the correlation is Pearson rather than rank
based, it survives to the physical injections only through affine transforms; a
nonlinear transform changes it. Rank correlation stays the package default for
that reason.

`nodes` is the number of Gauss-Hermite nodes per dimension. The default is
accurate to about 1e-8 for the usual load and generation marginals. Heavy-tailed
or strongly skewed marginals converge more slowly and want more nodes.

Skewed marginals cannot attain every correlation. A target outside the attainable
range throws, naming the pair and the range.
"""
function pearson_to_gaussian(
    rho::Real,
    di::UnivariateDistribution,
    dj::UnivariateDistribution;
    nodes::Integer = 32,
)
    -1 <= rho <= 1 || throw(ArgumentError("correlation must lie in [-1, 1], got $(rho)"))
    z, w = gauss_hermite(nodes)
    gi = standardized_marginal(di, 1)
    gj = standardized_marginal(dj, 2)
    return nataf_pair(Float64(rho), [gi(zk) for zk in z], gj, z, w, 1, 2)
end

function pearson_to_gaussian(
    R::AbstractMatrix{<:Real},
    marginals::AbstractVector;
    nodes::Integer = 32,
)
    dists = marginal_dists(marginals)
    d = size(R, 1)
    size(R, 2) == d ||
        throw(ArgumentError("correlation matrix must be square, got size $(size(R))"))
    issymmetric(R) || throw(ArgumentError("correlation matrix must be symmetric"))
    all(isapprox.(diag(R), 1.0; atol = 1e-12)) ||
        throw(ArgumentError("correlation matrix must have a unit diagonal"))
    length(dists) == d || throw(
        ArgumentError(
            "got $(length(dists)) marginals for a $(d) by $(d) correlation matrix",
        ),
    )
    z, w = gauss_hermite(nodes)
    g = [standardized_marginal(dists[k], k) for k = 1:d]
    g_at_nodes = [[gk(zk) for zk in z] for gk in g]
    Rg = Matrix{Float64}(I, d, d)
    for i = 1:d, j = (i+1):d
        rho0 = nataf_pair(Float64(R[i, j]), g_at_nodes[i], g[j], z, w, i, j)
        Rg[i, j] = rho0
        Rg[j, i] = rho0
    end
    return Rg
end

marginal_dists(v::AbstractVector{<:UnivariateDistribution}) = v
marginal_dists(v::AbstractVector{<:GermVariable}) = [x.dist for x in v]
marginal_dists(v::AbstractVector) = throw(
    ArgumentError(
        "marginals must be univariate distributions or GermVariables, got " *
        "element type $(eltype(v))",
    ),
)

"""
    GaussianCopula(R, marginals; correlation = :pearson, nodes = 32)

Gaussian copula whose parameter is the Nataf correction of the Pearson target `R`
for the given `marginals`, in germ variable order. Passing the germ variables
themselves keeps that order right:

```julia
vars = [GermVariable("a", LogNormal(0.0, 0.4)), GermVariable("b", Weibull(2.0, 8.0))]
dep = GaussianCopula([1.0 0.6; 0.6 1.0], vars; correlation = :pearson)
```

See [`pearson_to_gaussian`](@ref) for what the correction does, which targets are
attainable, and why the target is a property of the germ. Positive
semi-definiteness is validated after the correction, as for every other input
convention.
"""
function GaussianCopula(
    R::AbstractMatrix{<:Real},
    marginals::AbstractVector;
    correlation::Symbol = :pearson,
    nodes::Integer = 32,
)
    correlation === :pearson || throw(
        ArgumentError(
            "marginals are only used by correlation = :pearson, but got " *
            "$(repr(correlation)). Build the copula without them for :spearman " *
            "and :gaussian, which need no marginal information",
        ),
    )
    return GaussianCopula(pearson_to_gaussian(R, marginals; nodes); correlation = :gaussian)
end
