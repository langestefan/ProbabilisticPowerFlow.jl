@testmodule RankCorr begin
    using Statistics
    # Sample Spearman rank correlation of two vectors.
    ranks(x) = invperm(sortperm(x))
    spearman(x, y) = cor(float.(ranks(x)), float.(ranks(y)))
end

@testitem "spearman_to_gaussian map" tags = [:unit, :fast] begin
    @test spearman_to_gaussian(0.0) == 0.0
    @test spearman_to_gaussian(1.0) ≈ 1.0
    @test spearman_to_gaussian(-1.0) ≈ -1.0
    @test spearman_to_gaussian(0.5) ≈ 2 * sin(pi / 12)
end

@testitem "GaussianCopula construction and validation" tags = [:unit, :fast] begin
    using LinearAlgebra
    c = GaussianCopula([1.0 0.8; 0.8 1.0])
    @test c.L isa LowerTriangular
    @test size(c.L) == (2, 2)

    # Not a correlation matrix at all.
    @test_throws ArgumentError GaussianCopula([1.0 0.5 0.1; 0.5 1.0 0.2])
    @test_throws ArgumentError GaussianCopula([1.0 0.5; 0.4 1.0])
    @test_throws ArgumentError GaussianCopula([2.0 0.5; 0.5 2.0])

    # Valid diagonal but not PSD: the error must name the offending eigenvalue.
    R = [1.0 0.9 -0.9; 0.9 1.0 0.9; -0.9 0.9 1.0]
    err = try
        GaussianCopula(R)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("eigenvalue", err.msg)

    # Pearson requires the (unimplemented) Nataf correction: loud rejection,
    # never silent reinterpretation.
    @test_throws ArgumentError GaussianCopula([1.0 0.5; 0.5 1.0]; correlation = :pearson)
    @test_throws ArgumentError GaussianCopula([1.0 0.5; 0.5 1.0]; correlation = :typo)
end

@testitem "UncertaintyModel validation" tags = [:unit, :fast] begin
    using Distributions
    v = GermVariable("a", Normal())
    target = ComponentRef(:load, "3", :pd)

    # Assignment references an undeclared variable.
    @test_throws ArgumentError UncertaintyModel(
        [v],
        [Assignment("nope", target)],
        IndependentCopula(),
    )
    # Variable referenced by no assignment.
    @test_throws ArgumentError UncertaintyModel(
        [v, GermVariable("unused", Normal())],
        [Assignment("a", target)],
        IndependentCopula(),
    )
    # Duplicate ids.
    @test_throws ArgumentError UncertaintyModel(
        [v, GermVariable("a", Normal())],
        [Assignment("a", target)],
        IndependentCopula(),
    )
    # Copula dimension mismatch.
    @test_throws ArgumentError UncertaintyModel(
        [v],
        [Assignment("a", target)],
        GaussianCopula([1.0 0.0; 0.0 1.0]),
    )

    m = UncertaintyModel([v], [Assignment("a", target)], IndependentCopula())
    @test germ_dim(m) == 1
    @test targets(m) == [target]
end

@testitem "to_physical marginals and correlation" tags = [:unit, :fast] setup = [RankCorr] begin
    using Distributions, Random, Statistics
    d1 = Weibull(2.0, 8.5)
    d2 = Normal(1.0, 0.05)
    rho_s = 0.6
    m = UncertaintyModel(
        [GermVariable("wind", d1), GermVariable("load", d2)],
        [
            Assignment("wind", ComponentRef(:gen, "2", :pg)),
            Assignment("load", ComponentRef(:load, "3", :pd)),
        ],
        GaussianCopula([1.0 rho_s; rho_s 1.0]),
    )

    rng = Xoshiro(1234)
    n = 20_000
    out = Matrix{Float64}(undef, 2, n)
    for k = 1:n
        out[:, k] = to_physical(m, rand(rng, 2))
    end

    @test mean(out[1, :]) ≈ mean(d1) rtol = 0.02
    @test std(out[1, :]) ≈ std(d1) rtol = 0.05
    @test quantile(out[1, :], 0.9) ≈ quantile(d1, 0.9) rtol = 0.05
    @test mean(out[2, :]) ≈ mean(d2) rtol = 0.02

    # Spearman correlation survives the marginal transforms (rank invariance).
    @test RankCorr.spearman(out[1, :], out[2, :]) ≈ rho_s atol = 0.05
end

@testitem "independent copula stays independent" tags = [:unit, :fast] setup = [RankCorr] begin
    using Distributions, Random
    m = UncertaintyModel(
        [GermVariable("a", Normal()), GermVariable("b", Normal())],
        [
            Assignment("a", ComponentRef(:load, "3", :pd)),
            Assignment("b", ComponentRef(:load, "4", :pd)),
        ],
        IndependentCopula(),
    )
    rng = Xoshiro(99)
    n = 20_000
    out = Matrix{Float64}(undef, 2, n)
    for k = 1:n
        out[:, k] = to_physical(m, rand(rng, 2))
    end
    @test abs(RankCorr.spearman(out[1, :], out[2, :])) < 0.05
end

@testitem "pipeline is deterministic in u" tags = [:unit, :fast] begin
    using Distributions
    m = UncertaintyModel(
        [GermVariable("a", LogNormal(0.0, 0.3))],
        [Assignment("a", ComponentRef(:load, "3", :pd), AffineTransform(2.0, 1.0))],
        IndependentCopula(),
    )
    u = [0.73]
    @test to_physical(m, u) == to_physical(m, u)
    @test to_physical(m, u)[1] == 2.0 * quantile(LogNormal(0.0, 0.3), 0.73) + 1.0
    @test_throws DimensionMismatch to_physical(m, [0.5, 0.5])
end

@testitem "shared germ variable couples assignments" tags = [:unit, :fast] begin
    using Distributions, Random
    # P and Q of a constant-power-factor load share one germ variable: no
    # degenerate rho = 1 correlation entry needed.
    pf_ratio = 0.4  # qd = pf_ratio * pd
    m = UncertaintyModel(
        [GermVariable("load", Normal(0.5, 0.05))],
        [
            Assignment("load", ComponentRef(:load, "3", :pd), AffineTransform(1.0, 0.0)),
            Assignment(
                "load",
                ComponentRef(:load, "3", :qd),
                AffineTransform(pf_ratio, 0.0),
            ),
        ],
        IndependentCopula(),
    )
    rng = Xoshiro(7)
    for _ = 1:100
        x = to_physical(m, rand(rng, 1))
        @test x[2] ≈ pf_ratio * x[1]
    end
end
