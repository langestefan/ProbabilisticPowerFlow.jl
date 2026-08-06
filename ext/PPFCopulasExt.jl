module PPFCopulasExt

import Copulas
import ProbabilisticPowerFlow as PPF

PPF.dependence_dim(dep::PPF.CopulaDependence{<:Copulas.Copula}) = length(dep.copula)

function PPF.to_dependent!(
    v::AbstractVector{Float64},
    dep::PPF.CopulaDependence{<:Copulas.Copula},
    u::AbstractVector{<:Real},
)
    w = Copulas.inverse_rosenblatt(dep.copula, u)
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
