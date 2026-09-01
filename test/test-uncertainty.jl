@testsnippet Uncertainty begin
    using Distributions
    using Copulas
    using LinearAlgebra: cholesky
    using ProbabilisticPowerFlow: AffineTransform, IdentityTransform

    # wind in m/s and a load level
    VARIABLES =
        [GermVariable("wind", Weibull(2.0, 8.0)), GermVariable("load", Normal(1.0, 0.1))]
    ASSIGNMENTS = [
        Assignment("load", ComponentRef(ComponentField.Pd, 3), AffineTransform(0.50, 0.0)),
        Assignment("load", ComponentRef(ComponentField.Qd, 3), AffineTransform(0.15, 0.0)),
        Assignment("wind", ComponentRef(ComponentField.Pg, 7)),
    ]
    model(rho) =
        UncertaintyModel(VARIABLES, ASSIGNMENTS, GaussianCopula([1.0 rho; rho 1.0]))
end

@testitem "Transforms are values, not closures" tags=[:unit, :fast] begin
    using ProbabilisticPowerFlow: AffineTransform, IdentityTransform

    @test IdentityTransform()(3) === 3.0
    @test AffineTransform(2.0, 1.0)(3.0) === 7.0

    @test AffineTransform(2.0, 1.0) == AffineTransform(2.0, 1.0)
    @test AffineTransform(2.0, 1.0) != AffineTransform(2.0, 0.0)
    @test IdentityTransform() == IdentityTransform()
end

@testitem "varindex maps assignment order to germ order" setup=[Uncertainty] tags=[
    :unit,
    :fast,
] begin
    m = model(0.0)

    @test m.varindex == [2, 2, 1]
    @test germ_dim(m) == 2
    @test targets(m) == [a.target for a in ASSIGNMENTS]
    @test length(targets(m)) == length(m.assignments)
end

@testitem "The two-argument form defaults to independence" setup=[Uncertainty] tags=[
    :unit,
    :fast,
] begin
    m = UncertaintyModel(VARIABLES, ASSIGNMENTS)

    @test m.dependence isa IndependentCopula
    @test length(m.dependence) == germ_dim(m)
end

@testitem "Homogeneous marginals keep a concrete element type" tags=[:unit, :fast] begin
    using Distributions

    vars = [GermVariable("a", Normal()), GermVariable("b", Normal())]
    assigns = [
        Assignment("a", ComponentRef(ComponentField.Pd, 1)),
        Assignment("b", ComponentRef(ComponentField.Pd, 2)),
    ]
    @test eltype(UncertaintyModel(vars, assigns).variables) == GermVariable{Normal{Float64}}
end

@testitem "The constructor rejects an inconsistent model" setup=[Uncertainty] tags=[
    :unit,
    :fast,
] begin
    @test_throws ArgumentError UncertaintyModel([VARIABLES[1], VARIABLES[1]], ASSIGNMENTS)
    @test_throws ArgumentError UncertaintyModel(
        VARIABLES,
        [Assignment("typo", ComponentRef(ComponentField.Pd, 1))],
    )
    @test_throws ArgumentError UncertaintyModel(VARIABLES, ASSIGNMENTS[1:2])
    @test_throws ArgumentError UncertaintyModel(
        VARIABLES,
        ASSIGNMENTS,
        IndependentCopula(3),
    )
end

@testitem "to_physical is deterministic in u" setup=[Uncertainty] tags=[:unit, :fast] begin
    m = model(0.3)
    u = [0.30, 0.70]

    @test to_physical(m, u) == to_physical(m, u)
    @test to_physical(m, u) != to_physical(m, [0.31, 0.70])
end

@testitem "Independence and identity reduce to inverse transform sampling" tags=[
    :unit,
    :fast,
] begin
    using Distributions

    dists = [Weibull(2.0, 8.0), Normal(1.0, 0.1)]
    vars = [GermVariable("a", dists[1]), GermVariable("b", dists[2])]
    assigns = [
        Assignment("a", ComponentRef(ComponentField.Pg, 1)),
        Assignment("b", ComponentRef(ComponentField.Pd, 2)),
    ]
    m = UncertaintyModel(vars, assigns)
    u = [0.30, 0.70]

    # an independent copula is the identity on u, so x is exactly F^-1(u)
    @test to_physical(m, u) == [quantile(dists[k], u[k]) for k = 1:2]
end

@testitem "A Gaussian copula applies the Cholesky construction" setup=[Uncertainty] tags=[
    :unit,
    :fast,
] begin
    rho = 0.6
    m = model(rho)
    u = [0.30, 0.70]

    R = [1.0 rho; rho 1.0]
    dependent = cdf.(Normal(), cholesky(R).L * quantile.(Normal(), u))
    germ = [quantile(VARIABLES[k].dist, dependent[k]) for k = 1:2]
    expected = [0.50 * germ[2], 0.15 * germ[2], germ[1]]

    @test to_physical(m, u) ≈ expected
end

@testitem "A shared germ variable gives a constant ratio" setup=[Uncertainty] tags=[
    :unit,
    :fast,
] begin
    m = model(0.4)

    for u in ([0.1, 0.1], [0.5, 0.5], [0.9, 0.2], [0.02, 0.97])
        x = to_physical(m, u)
        @test x[2] / x[1] ≈ 0.15 / 0.50
    end
end

@testitem "The clamp keeps extreme uniforms finite" setup=[Uncertainty] tags=[:unit, :fast] begin
    vars = [GermVariable("a", Normal()), GermVariable("b", Normal())]
    assigns = [
        Assignment("a", ComponentRef(ComponentField.Pd, 1)),
        Assignment("b", ComponentRef(ComponentField.Pd, 2)),
    ]

    for C in (ClaytonCopula(2, 5.0), GaussianCopula([1.0 0.9; 0.9 1.0]))
        m = UncertaintyModel(vars, assigns, C)
        for u in ([1.0 - 1e-16, 1.0 - 1e-16], [1e-300, 0.5], [0.5, 0.5])
            @test all(isfinite, to_physical(m, u))
        end
    end
end

@testitem "to_physical! writes its buffers and names a bad one" setup=[Uncertainty] tags=[
    :unit,
    :fast,
] begin
    m = model(0.3)
    u = [0.30, 0.70]

    x = zeros(3)
    germ = zeros(2)
    @test to_physical!(x, m, u, germ) === x
    @test x == to_physical(m, u)

    @test x[3] == germ[1]           # identity transform on the wind assignment
    @test x[1] ≈ 0.50 * germ[2]     # affine transform on the load assignment
    @test all(>(0.0), germ)         # a wind speed and a load level

    @test_throws DimensionMismatch to_physical!(zeros(3), m, [0.3], zeros(2))
    @test_throws DimensionMismatch to_physical!(zeros(3), m, u, zeros(1))
    @test_throws DimensionMismatch to_physical!(zeros(2), m, u, zeros(2))

    err = try
        to_physical!(zeros(2), m, u, zeros(2))
    catch e
        sprint(showerror, e)
    end
    @test occursin("x has length 2", err)
end

@testitem "germ_dist is the joint distribution of the germ" setup=[Uncertainty] tags=[
    :unit,
    :fast,
] begin
    m = model(0.5)
    S = germ_dist(m)

    @test S isa SklarDist
    @test length(S) == germ_dim(m)
    @test S.m == (VARIABLES[1].dist, VARIABLES[2].dist)
    @test length(rand(S)) == germ_dim(m)
end

@testitem "Sampling reproduces the requested correlation" setup=[Uncertainty] tags=[
    :validation,
] begin
    using Random: Xoshiro

    rho = 0.6
    vars = [GermVariable("a", Normal()), GermVariable("b", Normal())]
    assigns = [
        Assignment("a", ComponentRef(ComponentField.Pd, 1)),
        Assignment("b", ComponentRef(ComponentField.Pd, 2)),
    ]
    m = UncertaintyModel(vars, assigns, GaussianCopula([1.0 rho; rho 1.0]))

    rng = Xoshiro(20260901)
    n = 20_000
    X = reduce(hcat, (to_physical(m, rand(rng, 2)) for _ = 1:n))

    # Gaussian margins under a Gaussian copula should return rho
    @test cor(X[1, :], X[2, :]) ≈ rho atol = 0.03
    @test mean(X[1, :]) ≈ 0.0 atol = 0.05
    @test std(X[1, :]) ≈ 1.0 atol = 0.05
end
