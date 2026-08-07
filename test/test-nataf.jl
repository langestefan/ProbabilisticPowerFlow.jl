@testmodule NatafRef begin
    using Distributions: LogNormal

    # Closed form for a lognormal pair: the Pearson correlation induced by a
    # Gaussian copula parameter rho0 is analytic, so the correction has an exact
    # answer to check against.
    lognormal_rho0(rho, s1, s2) =
        log(1 + rho * sqrt((exp(s1^2) - 1) * (exp(s2^2) - 1))) / (s1 * s2)
    lognormal_rho(rho0, s1, s2) =
        (exp(rho0 * s1 * s2) - 1) / sqrt((exp(s1^2) - 1) * (exp(s2^2) - 1))
end

@testitem "Gauss-Hermite rule integrates the normal moments" tags = [:unit, :fast] begin
    z, w = ProbabilisticPowerFlow.gauss_hermite(32)

    @test length(z) == length(w) == 32
    @test sum(w) ≈ 1.0
    @test sum(w .* z) ≈ 0.0 atol = 1e-14
    @test sum(w .* z .^ 2) ≈ 1.0
    @test sum(w .* z .^ 4) ≈ 3.0
    @test sum(w .* z .^ 6) ≈ 15.0
    # the rule is symmetric about zero
    @test z ≈ -reverse(z)
    @test w ≈ reverse(w)
    @test_throws ArgumentError ProbabilisticPowerFlow.gauss_hermite(1)
end

@testitem "Nataf correction reproduces the closed forms" tags = [:unit, :fast] setup =
    [NatafRef] begin
    using Distributions: LogNormal, Normal, Uniform

    # Gaussian marginals are the fixed point: the copula parameter is the target
    @test pearson_to_gaussian(0.7, Normal(3, 2), Normal(-1, 0.5)) ≈ 0.7 atol = 1e-10
    @test pearson_to_gaussian(-0.4, Normal(), Normal()) ≈ -0.4 atol = 1e-10

    # lognormal pairs have an analytic correction
    for (rho, s1, s2) in ((0.5, 0.4, 0.6), (-0.3, 0.25, 0.25), (0.8, 0.1, 0.15))
        got = pearson_to_gaussian(rho, LogNormal(0.0, s1), LogNormal(1.0, s2))
        @test got ≈ NatafRef.lognormal_rho0(rho, s1, s2) atol = 1e-9
    end

    # uniform marginals have equal Pearson and Spearman correlation, so the
    # correction must land on the closed-form rank map
    @test pearson_to_gaussian(0.6, Uniform(0, 1), Uniform(2, 5)) ≈ spearman_to_gaussian(0.6) atol =
        1e-9

    # zero stays exactly zero, so an uncorrelated pair keeps a structural zero
    @test pearson_to_gaussian(0.0, LogNormal(0.0, 0.5), Uniform(0, 1)) === 0.0
end

@testitem "Nataf correction converges in the number of nodes" tags = [:unit] setup =
    [NatafRef] begin
    using Distributions: LogNormal

    # a heavy right tail is the hard case for Gauss-Hermite, and the default node
    # count still lands on the analytic answer
    exact = NatafRef.lognormal_rho0(0.3, 0.2, 2.0)
    @test pearson_to_gaussian(0.3, LogNormal(0, 0.2), LogNormal(0, 2.0)) ≈ exact atol = 1e-8
    @test pearson_to_gaussian(0.3, LogNormal(0, 0.2), LogNormal(0, 2.0); nodes = 128) ≈
          exact atol = 1e-9
end

@testitem "Nataf correction of a full matrix" tags = [:unit, :fast] setup = [NatafRef] begin
    using Distributions: LogNormal, Normal, Weibull
    using LinearAlgebra: diag, issymmetric

    vars = [
        GermVariable("a", LogNormal(0.0, 0.4)),
        GermVariable("b", Weibull(2.0, 8.0)),
        GermVariable("c", Normal(2.0, 0.5)),
    ]
    R = [1.0 0.6 0.0; 0.6 1.0 0.2; 0.0 0.2 1.0]
    Rg = pearson_to_gaussian(R, vars)

    @test issymmetric(Rg)
    @test all(diag(Rg) .== 1.0)
    @test Rg[1, 3] === 0.0                  # structural zeros survive exactly
    @test Rg[1, 2] > R[1, 2]                # skewed marginals need a larger parameter
    @test Rg ≈ pearson_to_gaussian(R, [v.dist for v in vars])   # both input forms agree
    @test Rg ≈ pearson_to_gaussian(R, vars; nodes = 64) atol = 1e-9

    # a Gaussian block is untouched by the correction
    normals = [GermVariable("n$(k)", Normal(k, k)) for k = 1:3]
    @test pearson_to_gaussian(R, normals) ≈ R atol = 1e-9
end

@testitem "Nataf correction rejects bad input" tags = [:unit, :fast] begin
    using Distributions: Cauchy, LogNormal, Normal

    R = [1.0 0.3; 0.3 1.0]
    normals = [Normal(), Normal()]

    # no finite variance, so there is no Pearson correlation to match
    @test_throws ArgumentError pearson_to_gaussian(0.5, Cauchy(), Normal())
    @test_throws ArgumentError pearson_to_gaussian(1.5, Normal(), Normal())
    @test_throws ArgumentError pearson_to_gaussian(R, [Normal()])
    @test_throws ArgumentError pearson_to_gaussian(R, [1.0, 2.0])
    @test_throws ArgumentError pearson_to_gaussian([1.0 0.3; 0.4 1.0], normals)
    @test_throws ArgumentError pearson_to_gaussian([1.0 0.3; 0.3 0.9], normals)

    # marginals only mean anything for a Pearson target
    @test_throws ArgumentError GaussianCopula(R, normals; correlation = :spearman)
    @test_throws ArgumentError GaussianCopula(R, normals; correlation = :gaussian)

    # skewed marginals cannot attain every correlation, and the error says so
    err = try
        pearson_to_gaussian(0.5, LogNormal(0, 0.2), LogNormal(0, 2.0))
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("outside the range", err.msg)
    @test occursin("0.3325", err.msg)     # the analytic upper bound, rounded

    # PSD is still validated, on the corrected matrix
    bad = [1.0 0.9 -0.9; 0.9 1.0 0.9; -0.9 0.9 1.0]
    err2 = try
        GaussianCopula(bad, [Normal(), Normal(), Normal()])
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test occursin("positive semi-definite", err2.msg)
end

@testitem "a Pearson target without marginals points at the correction" tags =
    [:unit, :fast] begin
    err = try
        GaussianCopula([1.0 0.3; 0.3 1.0]; correlation = :pearson)
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("needs the marginals", err.msg)
    @test occursin("GaussianCopula(R, variables", err.msg)
end

@testitem "corrected samples hit the Pearson target" tags = [:unit] begin
    using Distributions: LogNormal, Normal, Weibull
    using Random: Xoshiro, rand!
    using Statistics: cor

    vars = [
        GermVariable("a", LogNormal(0.0, 0.4)),
        GermVariable("b", Weibull(2.0, 8.0)),
        GermVariable("c", Normal(2.0, 0.5)),
    ]
    assigns =
        [Assignment(v.id, ComponentRef(:load, "$(k)", :pd)) for (k, v) in enumerate(vars)]
    R = [1.0 0.6 -0.3; 0.6 1.0 0.2; -0.3 0.2 1.0]

    function germ_samples(dep, n)
        m = UncertaintyModel(vars, assigns, dep)
        rng = Xoshiro(20260807)
        X = Matrix{Float64}(undef, 3, n)
        u = Vector{Float64}(undef, 3)
        g = Vector{Float64}(undef, 3)
        for i = 1:n
            rand!(rng, u)
            to_physical!(view(X, :, i), m, u, g)
        end
        return cor(X, dims = 2)
    end

    n = 200_000
    corrected = germ_samples(GaussianCopula(R, vars; correlation = :pearson), n)
    for (i, j) in ((1, 2), (1, 3), (2, 3))
        @test corrected[i, j] ≈ R[i, j] atol = 0.01
    end
end

@testitem "the correction is what closes the gap on skewed marginals" tags = [:unit] begin
    using Distributions: LogNormal
    using Random: Xoshiro, rand!
    using Statistics: cor

    # mild skew makes the rank and Pearson conventions nearly agree, so the test
    # uses marginals skewed enough for the two to visibly part
    vars = [GermVariable("a", LogNormal(0.0, 1.2)), GermVariable("b", LogNormal(0.5, 1.2))]
    assigns =
        [Assignment(v.id, ComponentRef(:load, "$(k)", :pd)) for (k, v) in enumerate(vars)]
    R = [1.0 0.6; 0.6 1.0]

    function pearson_of(dep, n)
        m = UncertaintyModel(vars, assigns, dep)
        rng = Xoshiro(20260807)
        X = Matrix{Float64}(undef, 2, n)
        u = Vector{Float64}(undef, 2)
        g = Vector{Float64}(undef, 2)
        for i = 1:n
            rand!(rng, u)
            to_physical!(view(X, :, i), m, u, g)
        end
        return cor(X, dims = 2)[1, 2]
    end

    # heavy tails make the sample correlation itself noisy, hence the wide band
    corrected = pearson_of(GaussianCopula(R, vars; correlation = :pearson), 200_000)
    ranked = pearson_of(GaussianCopula(R), 200_000)
    @test corrected ≈ 0.6 atol = 0.04
    @test 0.6 - ranked > 0.1
end

@testitem "the corrected copula composes with the sampling methods" tags = [:integration] begin
    using Distributions: LogNormal, Normal
    using Random: Xoshiro
    using Statistics: mean

    vars = [
        GermVariable("p3", LogNormal(log(0.45), 0.15)),
        GermVariable("p5", Normal(0.60, 0.03)),
    ]
    assigns = [
        Assignment("p3", ComponentRef(:load, "3", :pd)),
        Assignment("p5", ComponentRef(:load, "5", :pd)),
    ]
    dep = GaussianCopula([1.0 0.5; 0.5 1.0], vars; correlation = :pearson)
    prob = PPFProblem(
        ReferenceBackend(case5()),
        UncertaintyModel(vars, assigns, dep),
        [VoltageMagnitude(5)],
    )

    mc = solve(prob, MonteCarlo(n = 400); rng = Xoshiro(4))
    lhs = solve(prob, LatinHypercube(n = 400); rng = Xoshiro(4))
    @test n_converged(mc) == 400
    @test n_converged(lhs) == 400
    @test mean(mc, VoltageMagnitude(5)) ≈ mean(lhs, VoltageMagnitude(5)) atol = 5e-4
end
