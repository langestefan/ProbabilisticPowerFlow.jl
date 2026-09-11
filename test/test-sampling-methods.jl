@testsnippet SamplingSetup begin
    stratified(U) =
        all(sort(floor.(Int, row .* size(U, 2))) == 0:(size(U, 2) - 1) for row in eachrow(U))

    function nominal_qois(prob, data)
        b = prob.backend
        state = init_state(b, targets(prob.model))
        set_injections!(state, b, [base_pd(data, bus) for bus in LOAD_BUSES])
        solve!(state, b)
        return [extract(state, b, q) for q in QOIS[1:3]]
    end
end

@testitem "Sampling methods reject a non-positive sample count" tags = [:unit, :fast] begin
    using QuasiMonteCarlo: SobolSample

    @test LatinHypercube().n == 1000
    @test QuasiMC(SobolSample()).n == 1024
    @test_throws ArgumentError LatinHypercube(n = 0)
    @test_throws ArgumentError QuasiMC(SobolSample(); n = 0)
end

@testitem "Latin hypercube points fill every stratum once" tags = [:unit, :fast] setup =
    [SamplingSetup] begin
    using Random: Xoshiro

    U = ProbabilisticPowerFlow.lhs_points(Xoshiro(1), 4, 50)
    @test size(U) == (4, 50)
    @test all(0 .< U .< 1)
    @test stratified(U)
    @test U == ProbabilisticPowerFlow.lhs_points(Xoshiro(1), 4, 50)
end

@testitem "QuasiMC needs a QuasiMonteCarlo sampler" tags = [:unit, :fast] begin
    using QuasiMonteCarlo: QuasiMonteCarlo

    @test_throws MethodError QuasiMC(:halton; n = 8)
end

@testitem "Every sampler agrees with a deterministic solve at nominal loading" tags =
    [:integration, :powermodels] setup = [PMCase5, MCCase5, SamplingSetup] begin
    using QuasiMonteCarlo: SobolSample, HaltonSample

    data = pm_case5()
    prob = load_problem(data)
    reference = nominal_qois(prob, data)

    results = [
        solve(prob, LatinHypercube(n = 512); rng = Xoshiro(1)),
        solve(prob, QuasiMC(SobolSample(); n = 512)),
        solve(prob, QuasiMC(HaltonSample(); n = 512)),
    ]
    for r in results
        @test r.n_samples == 512
        @test r.n_solves == 512
        @test failure_rate(r) == 0
        @test 0 < violation_probability(r, BAND) < 1
        for (q, ref) in zip(QOIS[1:3], reference)
            @test mean(r, q) ≈ ref rtol = 0.01 atol = 1.0e-3
        end
    end
end

@testitem "Randomized samplers follow their seed" tags = [:integration, :powermodels] setup =
    [PMCase5, MCCase5] begin
    using QuasiMonteCarlo: SobolSample, OwenScramble

    prob = load_problem(pm_case5())
    r1 = solve(prob, LatinHypercube(n = 32); rng = Xoshiro(7))
    r2 = solve(prob, LatinHypercube(n = 32); rng = Xoshiro(7))
    r3 = solve(prob, LatinHypercube(n = 32); rng = Xoshiro(8))
    @test r1.samples == r2.samples
    @test r1.samples != r3.samples

    scrambled(seed) =
        QuasiMC(SobolSample(R = OwenScramble(base = 2, pad = 32, rng = Xoshiro(seed))); n = 32)
    @test solve(prob, scrambled(1)).samples == solve(prob, scrambled(1)).samples
    @test solve(prob, scrambled(1)).samples != solve(prob, scrambled(2)).samples
end

@testitem "LHS and shifted Sobol scatter less than MonteCarlo" tags =
    [:integration, :powermodels] setup = [PMCase5, MCCase5] begin
    using QuasiMonteCarlo: SobolSample, Shift

    prob = load_problem(pm_case5())
    seeds = 101:112
    mc = [mean(solve(prob, MonteCarlo(n = 64); rng = Xoshiro(s)), VM5) for s in seeds]
    lhs = [mean(solve(prob, LatinHypercube(n = 64); rng = Xoshiro(s)), VM5) for s in seeds]
    sobol = [
        mean(solve(prob, QuasiMC(SobolSample(R = Shift(rng = Xoshiro(s))); n = 64)), VM5)
            for s in seeds
    ]

    @test std(lhs) < std(mc)
    @test std(sobol) < std(mc)
end

@testitem "Warm start and scheduler options are validated" tags = [:unit, :fast] begin
    using Distributions: Normal

    struct ColdBackend <: AbstractPFBackend end
    ref = ComponentRef(ComponentField.Pd, 1)
    model = UncertaintyModel([GermVariable("a", Normal())], [Assignment("a", ref)])
    prob = PPFProblem(ColdBackend(), model, [VoltageMagnitude(1)])

    @test_throws ArgumentError solve(prob, MonteCarlo(n = 2); warmstart = :backwards)
    @test_throws ArgumentError solve(prob, MonteCarlo(n = 2); warmstart = :chain)
    @test_throws MethodError solve(prob, MonteCarlo(n = 2); scheduler = :threads)
end

@testitem "Warm starts and schedulers change the cost, not the result" tags =
    [:integration, :powermodels] setup = [PMCase5, MCCase5] begin
    using OhMyThreads: DynamicScheduler, StaticScheduler

    prob = load_problem(pm_case5())
    method = MonteCarlo(n = 300)
    cold = solve(prob, method; rng = Xoshiro(11))

    for warmstart in (:chain, :sorted), scheduler in (nothing, DynamicScheduler())
        r = solve(prob, method; rng = Xoshiro(11), warmstart, scheduler)
        @test r.sample_indices == cold.sample_indices
        @test r.n_solves == 300
        @test isapprox(r.samples, cold.samples; atol = 1.0e-6)
    end

    warm = solve(prob, method; rng = Xoshiro(11), scheduler = StaticScheduler())
    @test warm.samples == cold.samples
end

@testitem "A scheduler records the same failures as a serial run" tags =
    [:integration, :powermodels] setup = [PMCase5, MCCase5] begin
    using OhMyThreads: DynamicScheduler

    prob = load_problem(pm_case5(); scale = 5.0, rel = 0.3)
    serial = solve(prob, MonteCarlo(n = 200); rng = Xoshiro(1))
    parallel =
        solve(prob, MonteCarlo(n = 200); rng = Xoshiro(1), scheduler = DynamicScheduler())

    @test !isempty(parallel.failures)
    @test [f.index for f in parallel.failures] == [f.index for f in serial.failures]
    @test parallel.samples == serial.samples

    chained = solve(
        prob,
        MonteCarlo(n = 200);
        rng = Xoshiro(1),
        warmstart = :chain,
        scheduler = DynamicScheduler(),
    )
    @test n_converged(chained) + length(chained.failures) == 200
    @test issorted([f.index for f in chained.failures])
    @test issorted(chained.sample_indices)
end
