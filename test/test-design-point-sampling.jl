@testsnippet ISCase5 begin
    using Distributions: Normal, Weibull
    using Copulas: GaussianCopula
    using Random: Xoshiro
    using Statistics: mean, std

    const IS_BUSES = 3:5
    const IS_VM = VoltageMagnitude(5)

    # Normal loads at buses 3 to 5 with correlation 0.5, std rel * mean
    function is_problem(data, qois; rel = 0.1)
        vars = [GermVariable("pd$(b)", Normal(base_pd(data, b), rel * base_pd(data, b))) for b in IS_BUSES]
        assigns = [
            Assignment("pd$(b)", ComponentRef(ComponentField.Pd, load_at(data, b))) for b in IS_BUSES
        ]
        corr = [
            1.0 0.5 0.5
            0.5 1.0 0.5
            0.5 0.5 1.0
        ]
        model = UncertaintyModel(vars, assigns, GaussianCopula(corr))
        return PPFProblem(PowerModelsBackend(data), model, qois)
    end

    # the voltage at bus 5 at the median injections
    function median_vm(prob)
        g = ProbabilisticPowerFlow.LimitState(prob, ViolationEvent(IS_VM, -Inf, 0.0))
        return -g(zeros(germ_dim(prob.model)))
    end
end

@testitem "DesignPointSampling validates its arguments" tags = [:unit, :fast] begin
    over = ViolationEvent(VoltageMagnitude(1), -Inf, 1.1)
    m = DesignPointSampling(over)
    @test m.n == 1000
    @test m.shift === nothing
    @test DesignPointSampling(over; shift = [1, 2]).shift == [1.0, 2.0]

    @test_throws ArgumentError DesignPointSampling(over; n = 0)
    @test_throws ArgumentError DesignPointSampling(over; maxiter = 0)
    @test_throws ArgumentError DesignPointSampling(over; fd_step = 0)
    @test_throws ArgumentError DesignPointSampling(over; tol = -1.0)
    @test_throws ArgumentError DesignPointSampling(ViolationEvent(VoltageMagnitude(1), -Inf, Inf))

    @test sprint(show, m) == "DesignPointSampling(ViolationEvent(VoltageMagnitude(1), -Inf, 1.1), n = 1000)"
    @test sprint(show, DesignPointSampling(over; shift = [3.0, 4.0])) ==
        "DesignPointSampling(ViolationEvent(VoltageMagnitude(1), -Inf, 1.1), n = 1000, β = 5.0)"
end

@testitem "Weighted results expose estimator diagnostics" tags = [:unit, :fast] begin
    using Statistics: mean, std, quantile

    struct NoMethod <: AbstractPPFMethod end
    vm = VoltageMagnitude(2)
    band = ViolationEvent(vm, -Inf, 1.05)
    samples = [1.0 1.06 1.07]
    w = [2.0, 0.5, 0.1]
    r = PPFResult(NoMethod(), AbstractQoI[vm], samples, [1, 2, 3], FailedSample[], 3, 3, w)

    @test violation_probability(r, band) ≈ (0.5 + 0.1) / 3
    @test mean(r, vm) ≈ (2.0 + 0.53 + 0.107) / 3
    @test standard_error(r, band) ≈ std([0.0, 0.5, 0.1]) / sqrt(3)
    @test effective_sample_size(r, band) ≈ 0.6^2 / 0.26
    @test_throws ArgumentError std(r, vm)
    @test_throws ArgumentError quantile(r, vm, 0.5)
    @test occursin("weights: importance weights", sprint(show, MIME"text/plain"(), r))

    plain = PPFResult(NoMethod(), AbstractQoI[vm], samples, [1, 2, 3], FailedSample[], 3, 3)
    @test plain.weights === nothing
    @test effective_sample_size(plain, band) == 2
    @test standard_error(plain, vm) ≈ std(vec(samples)) / sqrt(3)
    @test !occursin("weights", sprint(show, MIME"text/plain"(), plain))
end

@testitem "DesignPointSampling needs the event among the QoIs" tags =
    [:integration, :powermodels] setup = [PMCase5, ISCase5] begin
    prob = is_problem(pm_case5(), [VoltageAngle(5)])
    over = ViolationEvent(IS_VM, -Inf, 1.1)
    @test_throws ArgumentError solve(prob, DesignPointSampling(over; n = 10))

    prob = is_problem(pm_case5(), [IS_VM])
    @test_throws DimensionMismatch solve(prob, DesignPointSampling(over; n = 10, shift = [1.0]))
end

@testitem "A zero shift is plain Monte Carlo in normal space" tags =
    [:integration, :powermodels] setup = [PMCase5, ISCase5] begin
    prob = is_problem(pm_case5(), [IS_VM])
    event = ViolationEvent(IS_VM, -Inf, median_vm(prob))
    r = solve(prob, DesignPointSampling(event; n = 400, shift = zeros(3)); rng = Xoshiro(3))

    @test r.n_solves == 400
    @test all(==(1.0), r.weights)
    @test effective_sample_size(r, event) == sum(qoi_samples(r, event))
    @test violation_probability(r, event) == mean(qoi_samples(r, event))
    # the event cuts the distribution at its median
    @test violation_probability(r, event) ≈ 0.5 atol = 0.1
end

@testitem "The design point lies on the limit state surface" tags =
    [:integration, :powermodels] setup = [PMCase5, ISCase5] begin
    prob = is_problem(pm_case5(), [IS_VM])
    v0 = median_vm(prob)
    for event in (ViolationEvent(IS_VM, -Inf, v0 + 0.004), ViolationEvent(IS_VM, v0 - 0.004, Inf))
        zstar, n_search = ProbabilisticPowerFlow.design_point(prob, event; tol = 1.0e-6)
        g = ProbabilisticPowerFlow.LimitState(prob, event)
        @test abs(g(zstar)) < 1.0e-6
        @test n_search > 0
        # moving along z* from the origin reaches the surface no sooner
        @test g(0.99 .* zstar) > 0
    end
end

@testitem "DesignPointSampling agrees with Monte Carlo and beats its variance" tags =
    [:integration, :powermodels] setup = [PMCase5, ISCase5] begin
    data = pm_case5()
    v0 = median_vm(is_problem(data, [IS_VM]))
    event = ViolationEvent(IS_VM, -Inf, v0 + 0.006)
    prob = is_problem(data, [IS_VM, event])

    mc = solve(prob, MonteCarlo(n = 40_000); rng = Xoshiro(1))
    p_mc = violation_probability(mc, event)
    se_mc = standard_error(mc, event)
    @test 0.001 < p_mc < 0.05

    r = solve(prob, DesignPointSampling(event; n = 2000); rng = Xoshiro(2))
    p_is = violation_probability(r, event)
    se_is = standard_error(r, event)

    @test r.n_samples == 2000
    @test r.n_solves > 2000
    @test r.method.shift !== nothing
    @test abs(p_is - p_mc) < 4 * sqrt(se_is^2 + se_mc^2)
    # with 20 times fewer samples, the estimate is still more precise
    @test se_is < se_mc
    # about half of the shifted samples hit the event, with similar weights
    hits = sum(qoi_samples(r, event))
    @test 0.3 < hits / n_converged(r) < 0.7
    @test effective_sample_size(r, event) > hits / 4

    # the stored design point skips the search
    again = solve(prob, r.method; rng = Xoshiro(2))
    @test again.n_solves == 2000
    @test again.samples == r.samples
    @test again.weights == r.weights
end

@testitem "DesignPointSampling handles non-Gaussian marginals and transforms" tags =
    [:integration, :powermodels] setup = [PMCase5, ISCase5] begin
    using Copulas: ClaytonCopula

    data = pm_case5()
    # Weibull "wind speed" scaled into a negative load at bus 4, a Normal load at bus 3
    # and a Clayton copula between them
    vars = [
        GermVariable("pd3", Normal(base_pd(data, 3), 0.1 * base_pd(data, 3))),
        GermVariable("wind", Weibull(2.0, 8.0)),
    ]
    assigns = [
        Assignment("pd3", ComponentRef(ComponentField.Pd, load_at(data, 3))),
        Assignment(
            "wind",
            ComponentRef(ComponentField.Pd, load_at(data, 4)),
            AffineTransform(-0.05 * base_pd(data, 4), base_pd(data, 4)),
        ),
    ]
    model = UncertaintyModel(vars, assigns, ClaytonCopula(2, 1.5))
    prob = PPFProblem(PowerModelsBackend(data), model, AbstractQoI[IS_VM])

    v0 = median_vm(prob)
    event = ViolationEvent(IS_VM, -Inf, v0 + 0.0015)
    prob = PPFProblem(PowerModelsBackend(data), model, AbstractQoI[IS_VM, event])

    mc = solve(prob, MonteCarlo(n = 40_000); rng = Xoshiro(1))
    # the Clayton copula makes plain HL-RF oscillate, the line search must settle it
    r = @test_logs solve(prob, DesignPointSampling(event; n = 2000); rng = Xoshiro(2))
    p_mc, se_mc = violation_probability(mc, event), standard_error(mc, event)
    p_is, se_is = violation_probability(r, event), standard_error(r, event)

    @test 0.001 < p_mc < 0.05
    @test abs(p_is - p_mc) < 4 * sqrt(se_is^2 + se_mc^2)
    @test se_is < se_mc
end
