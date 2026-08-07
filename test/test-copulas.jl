@testitem "a Copulas copula works directly as the dependence" tags = [:unit, :fast] begin
    import Copulas
    import ProbabilisticPowerFlow as PPF
    using Distributions: Normal

    C = Copulas.ClaytonCopula(2, 2.0)
    @test PPF.dependence_dim(C) == 2

    vars = [GermVariable("a", Normal()), GermVariable("b", Normal())]
    assigns = [
        Assignment("a", ComponentRef(:load, "3", :pd)),
        Assignment("b", ComponentRef(:load, "5", :pd)),
    ]
    @test UncertaintyModel(vars, assigns, C) isa UncertaintyModel
    # dimension mismatch is caught at model construction
    @test_throws ArgumentError UncertaintyModel(
        vars,
        assigns,
        Copulas.ClaytonCopula(3, 2.0),
    )

    # a type without the dependence interface is refused at model construction
    # with an error that points at the interface, not a bare MethodError
    err = try
        UncertaintyModel(vars, assigns, [1.0 0.0; 0.0 1.0])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("dependence interface", err.msg)
end

@testitem "a direct copula dependence is deterministic in u" tags = [:unit, :fast] begin
    import Copulas
    using Distributions: Weibull, Normal

    mk() = UncertaintyModel(
        [GermVariable("w", Weibull(2.0, 8.0)), GermVariable("l", Normal(1.0, 0.1))],
        [
            Assignment("w", ComponentRef(:load, "3", :pd)),
            Assignment("l", ComponentRef(:load, "5", :pd)),
        ],
        Copulas.GumbelCopula(2, 1.7),
    )
    m1 = mk()
    u = [0.31, 0.77]
    @test to_physical(m1, u) == to_physical(m1, u)
    # a fresh identical copula gives the same map
    @test to_physical(m1, u) == to_physical(mk(), u)
end

@testitem "the copula seam clamps extreme inputs" tags = [:unit, :fast] begin
    import Copulas
    import ProbabilisticPowerFlow as PPF

    # the upstream inverse Rosenblatt returns exactly 0.0 and even Inf at these
    # inputs, which the extension must clamp so marginal quantiles stay finite
    dep = Copulas.ClaytonCopula(2, 2.0)
    v = zeros(2)
    for u in ([eps(), 1 - eps()], [1e-300, 1 - eps() / 2])
        PPF.to_dependent!(v, dep, u)
        @test all(isfinite, v)
        @test all(0 .< v .< 1)
    end
end

@testitem "Clayton rank correlation and marginals" tags = [:unit] setup = [RankCorr] begin
    import Copulas
    import ProbabilisticPowerFlow as PPF
    using Distributions: Normal, Weibull, mean, quantile
    using Random: Xoshiro, rand!
    using Statistics

    C = Copulas.ClaytonCopula(2, 2.0)
    m = UncertaintyModel(
        [GermVariable("w", Weibull(2.0, 8.0)), GermVariable("l", Normal(1.0, 0.1))],
        [
            Assignment("w", ComponentRef(:load, "3", :pd)),
            Assignment("l", ComponentRef(:load, "5", :pd)),
        ],
        C,
    )

    rng = Xoshiro(7)
    n = 20_000
    X = Matrix{Float64}(undef, 2, n)
    u = Vector{Float64}(undef, 2)
    germ = Vector{Float64}(undef, 2)
    for i = 1:n
        rand!(rng, u)
        to_physical!(view(X, :, i), m, u, germ)
    end

    # the copula sets the rank correlation and the marginals stay intact
    @test RankCorr.spearman(X[1, :], X[2, :]) ≈ Copulas.ρ(C) atol = 0.05
    @test mean(X[1, :]) ≈ mean(Weibull(2.0, 8.0)) rtol = 0.05
    @test mean(X[2, :]) ≈ 1.0 atol = 0.01
    @test quantile(X[2, :], 0.5) ≈ 1.0 atol = 0.01
end

@testitem "Gaussian family matches the built-in copula pointwise" tags = [:unit] setup =
    [RankCorr] begin
    import Copulas
    import ProbabilisticPowerFlow as PPF
    using Random: Xoshiro, rand!

    # the inverse Rosenblatt of a Gaussian copula in coordinate order is exactly
    # the Cholesky map the built-in GaussianCopula implements, so agreement is
    # pointwise, limited only by the built-in 1e-12 Cholesky jitter
    Rg = [1.0 0.6; 0.6 1.0]
    from_copulas = Copulas.GaussianCopula(Rg)
    builtin = GaussianCopula(Rg; correlation = :gaussian)

    rng = Xoshiro(11)
    u = Vector{Float64}(undef, 2)
    vw = Vector{Float64}(undef, 2)
    vb = Vector{Float64}(undef, 2)
    for _ = 1:200
        rand!(rng, u)
        PPF.to_dependent!(vw, from_copulas, u)
        PPF.to_dependent!(vb, builtin, u)
        @test isapprox(vw, vb; atol = 1e-5)
    end
end

@testitem "MC end-to-end with a Clayton copula on case5" tags = [:integration] begin
    import Copulas
    using Distributions: Normal
    using Random: Xoshiro
    using Statistics: mean

    model = UncertaintyModel(
        [GermVariable("p3", Normal(0.45, 0.05)), GermVariable("p5", Normal(0.60, 0.06))],
        [
            Assignment("p3", ComponentRef(:load, "3", :pd)),
            Assignment("p5", ComponentRef(:load, "5", :pd)),
        ],
        Copulas.ClaytonCopula(2, 2.0),
    )
    prob = PPFProblem(ReferenceBackend(case5()), model, [VoltageMagnitude(5)])
    r = solve(prob, MonteCarlo(n = 500); rng = Xoshiro(3))
    @test n_converged(r) == 500
    @test 0.9 < mean(r, VoltageMagnitude(5)) < 1.1
end

@testitem "QMC composes with a Clayton copula" tags = [:integration] begin
    import Copulas
    using Sobol: Sobol
    using Distributions: Normal
    using Random: Xoshiro
    using Statistics: mean

    model = UncertaintyModel(
        [GermVariable("p3", Normal(0.45, 0.05)), GermVariable("p5", Normal(0.60, 0.06))],
        [
            Assignment("p3", ComponentRef(:load, "3", :pd)),
            Assignment("p5", ComponentRef(:load, "5", :pd)),
        ],
        Copulas.ClaytonCopula(2, 2.0),
    )
    prob = PPFProblem(ReferenceBackend(case5()), model, [VoltageMagnitude(5)])

    # determinism end to end: same seed, same samples
    r1 = solve(prob, SobolSampling(n = 256); rng = Xoshiro(5))
    r2 = solve(prob, SobolSampling(n = 256); rng = Xoshiro(5))
    @test r1.samples == r2.samples

    rmc = solve(prob, MonteCarlo(n = 2000); rng = Xoshiro(6))
    @test mean(r1, VoltageMagnitude(5)) ≈ mean(rmc, VoltageMagnitude(5)) atol = 0.002
end
