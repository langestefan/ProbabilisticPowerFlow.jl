@testitem "running_estimate follows the draws and skips a failure" tags = [:unit, :fast] begin
    using Statistics: mean

    vm = VoltageMagnitude(2)
    samples = [1.0 0.94 1.06]
    failure = FailedSample(3, [0.5], [2.0], SolveInfo(false, -1, Inf))
    r = PPFResult(MonteCarlo(n = 4), AbstractQoI[vm], samples, [1, 2, 4], [failure], 4, 4)
    run = running_estimate(r, vm)

    @test run.samples == 1:4
    @test run.solves == 1:4
    @test run.estimate ≈ [1.0, 0.97, 0.97, 1.0]
    @test isnan(run.standard_error[1])
    @test run.standard_error[3] == run.standard_error[2]
    @test run.estimate[end] ≈ mean(r, vm)
    @test run.standard_error[end] ≈ standard_error(r, vm)
end

@testitem "running_estimate of a weighted result counts a failure as zero" tags =
    [:unit, :fast] begin
    struct NoMethod <: AbstractPPFMethod end
    ProbabilisticPowerFlow.iid_draws(::NoMethod) = true

    event = ViolationEvent(VoltageMagnitude(2), -Inf, 1.05)
    failure = FailedSample(2, [0.5], [2.0], SolveInfo(false, -1, Inf))
    w = [0.5, 4.0, 0.1]
    r = PPFResult(NoMethod(), AbstractQoI[event.qoi], [1.06 1.07], [1, 3], [failure], 3, 3, w)
    run = running_estimate(r, event)

    @test run.estimate ≈ [0.5, 0.25, 0.2]
    @test run.estimate[end] ≈ violation_probability(r, event)
    @test run.standard_error[end] ≈ standard_error(r, event)
end

@testitem "running_estimate is NaN until a draw converges" tags = [:unit, :fast] begin
    vm = VoltageMagnitude(2)
    failure = FailedSample(1, [0.5], [2.0], SolveInfo(false, -1, Inf))
    r = PPFResult(MonteCarlo(n = 3), AbstractQoI[vm], [1.0 0.9], [2, 3], [failure], 3, 3)
    run = running_estimate(r, vm)

    @test isnan(run.estimate[1])
    @test run.estimate[2:3] ≈ [1.0, 0.95]
    @test isnan(run.standard_error[2])
    @test !isnan(run.standard_error[3])
end

@testitem "running_estimate needs independent draws" tags = [:unit, :fast] begin
    vm = VoltageMagnitude(2)
    stratified(method) = PPFResult(method, AbstractQoI[vm], [1.0 0.9], [1, 2], FailedSample[], 2, 2)

    @test_throws ArgumentError running_estimate(stratified(LatinHypercube(n = 2)), vm)
    @test_throws "LatinHypercube" running_estimate(stratified(LatinHypercube(n = 2)), vm)
end

@testitem "running_estimate of DesignPointSampling starts after the search" tags =
    [:unit, :fast] setup = [LinearOracle] begin
    event = ViolationEvent(ORACLE_Y, -Inf, 4.0)
    r = solve(oracle(event), DesignPointSampling(event; n = 500); rng = Xoshiro(1))
    run = running_estimate(r, event)

    @test r.n_setup_solves > 0
    @test r.n_setup_solves + r.n_samples == r.n_solves
    @test run.solves[1] == r.n_setup_solves + 1
    @test run.solves[end] == r.n_solves
    @test run.estimate[end] ≈ violation_probability(r, event)
    @test run.standard_error[end] ≈ standard_error(r, event)
end

@testitem "AdaptiveDesignPointSampling draws its mixture components at random" tags =
    [:unit, :fast] setup = [LinearOracle] begin
    event = ViolationEvent(ORACLE_Y, -3.5, 4.0)
    base = oracle(event)
    prob = PPFProblem(base.backend, base.model, [ORACLE_Y, event])
    r = @test_logs solve(prob, AdaptiveDesignPointSampling(event; n = 1000); rng = Xoshiro(1))
    run = running_estimate(r, event)

    # the first draws already come from around both design points
    first_draws = qoi_samples(r, ORACLE_Y)[1:50]
    @test any(<(0), first_draws)
    @test any(>(0), first_draws)

    @test run.solves[1] == r.n_setup_solves + 1
    @test run.solves[end] == r.n_solves
    @test run.estimate[end] ≈ violation_probability(r, event)
    @test run.standard_error[end] ≈ standard_error(r, event)
end
