module PPFCopulasExt

import Copulas
import ProbabilisticPowerFlow as PPF

# Any Copulas.jl copula works directly as the dependence of an UncertaintyModel.
# The extension only adds methods to the two duck-typed seam functions for the
# foreign type. No wrapper exists and core carries no Copulas-specific code.

PPF.dependence_dim(C::Copulas.Copula) = length(C)

function PPF.to_dependent!(
    v::AbstractVector{Float64},
    C::Copulas.Copula,
    u::AbstractVector{<:Real},
)
    w = Copulas.inverse_rosenblatt(C, u)
    # The clamp is mandatory. Archimedean families return exactly 0.0 or Inf at
    # extreme inputs and the empirical copula can return exactly 1.0, while the
    # marginal quantiles in step 2 of to_physical! must stay finite. clamp maps
    # Inf to 1 - eps(), so it also covers the non-finite escape.
    @inbounds for k in eachindex(v)
        v[k] = clamp(w[k], eps(), 1 - eps())
    end
    return v
end

end
