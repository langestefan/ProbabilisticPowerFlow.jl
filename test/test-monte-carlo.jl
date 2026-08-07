@testsnippet MCSetup begin
    using Distributions, Random, Statistics

    # Normal load uncertainty on the three PQ buses of case5, with the loads at
    # buses 3 and 4 rank-correlated.
    function case5_problem(; load_scale = 1.0, std_frac = 0.1)
        base_pd = load_scale .* [0.45, 0.40, 0.60]
        vars = [
            GermVariable("load$(bus)", Normal(base_pd[k], std_frac * base_pd[k])) for
            (k, bus) in enumerate(3:5)
        ]
        assigns =
            [Assignment("load$(bus)", ComponentRef(:load, "$(bus)", :pd)) for bus = 3:5]
        rho_s = 0.5
        R = [1.0 rho_s 0.0; rho_s 1.0 0.0; 0.0 0.0 1.0]
        model = UncertaintyModel(vars, assigns, GaussianCopula(R))
        qois = [
            VoltageMagnitude(5),
            VoltageAngle(3),
            ViolationEvent(VoltageMagnitude(5), 0.95, 1.05),
        ]
        return PPFProblem(ReferenceBackend(case5(; load_scale)), model, qois)
    end
end

@testitem "MC end-to-end on case5" tags = [:integration] setup = [MCSetup] begin
    prob = case5_problem()
    method = MonteCarlo(n = 2000)
    r = solve(prob, method; rng = Xoshiro(42))

    # Shape invariants.
    @test r.n_samples == 2000
    @test r.n_solves == 2000  # the benchmark cost currency: one solve per sample
    @test n_converged(r) + length(r.failures) == r.n_samples
    @test size(r.samples) == (3, n_converged(r))
    @test length(r.sample_indices) == n_converged(r)
    @test issorted(r.sample_indices)
    @test isempty(r.failures)  # nominal loading: everything converges
    @test failure_rate(r) == 0.0

    # Statistics: the MC mean must sit near the deterministic solve at the mean
    # injections (loose tolerance: the pipeline is nonlinear).
    state = init_state(prob.backend, ComponentRef[])
    solve!(state, prob.backend)
    v5_det = extract(state, prob.backend, VoltageMagnitude(5))
    @test mean(r, VoltageMagnitude(5)) ≈ v5_det atol = 0.005
    @test std(r, VoltageMagnitude(5)) > 0
    @test quantile(r, VoltageMagnitude(5), 0.05) < mean(r, VoltageMagnitude(5))

    v = ViolationEvent(VoltageMagnitude(5), 0.95, 1.05)
    @test 0.0 <= violation_probability(r, v) <= 1.0

    # Unknown QoI errors instead of silently returning garbage.
    @test_throws ArgumentError mean(r, VoltageMagnitude(4))
end

@testitem "MC is reproducible under a seed" tags = [:integration] setup = [MCSetup] begin
    prob = case5_problem()
    r1 = solve(prob, MonteCarlo(n = 500); rng = Xoshiro(7))
    r2 = solve(prob, MonteCarlo(n = 500); rng = Xoshiro(7))
    @test r1.samples == r2.samples
    @test r1.sample_indices == r2.sample_indices
end

@testitem "MC records failures as outputs" tags = [:integration] setup = [MCSetup] begin
    # Push the case toward the loadability nose so that a fraction of the samples
    # genuinely diverges; failure handling must record them, not drop or throw.
    prob = case5_problem(load_scale = 4.4, std_frac = 0.15)
    r = solve(prob, MonteCarlo(n = 400); rng = Xoshiro(2026))

    @test !isempty(r.failures)
    @test n_converged(r) > 0  # statistics still computed from the converged part
    @test n_converged(r) + length(r.failures) == r.n_samples
    @test r.n_solves == r.n_samples  # failed solves count toward the budget
    for fs in r.failures
        @test 1 <= fs.index <= r.n_samples
        @test length(fs.u) == 3
        @test all(0.0 .< fs.u .< 1.0)
        @test length(fs.injections) == 3  # the injection vector that caused it
        @test !fs.info.converged
    end
    @test mean(r, VoltageMagnitude(5)) < 1.0  # heavily loaded, depressed voltages

    # The :retry policy is reserved and not implemented, so it must throw.
    @test_throws ArgumentError solve(
        PPFProblem(prob.backend, prob.model, prob.qois),
        MonteCarlo(n = 10, failure_policy = :retry),
    )
end

@testitem "warm start scheduling changes cost, not results" tags = [:integration] setup =
    [MCSetup] begin
    prob = case5_problem()
    r_off = solve(prob, MonteCarlo(n = 500, warmstart = :off); rng = Xoshiro(11))
    r_chain = solve(prob, MonteCarlo(n = 500, warmstart = :chain); rng = Xoshiro(11))
    r_sorted = solve(prob, MonteCarlo(n = 500, warmstart = :sorted); rng = Xoshiro(11))

    # every mode solves the same 500 samples once
    @test r_off.n_solves == r_chain.n_solves == r_sorted.n_solves == 500
    @test isempty(r_off.failures) && isempty(r_chain.failures) && isempty(r_sorted.failures)

    # chained starts change the iteration path, not the solution
    @test r_chain.sample_indices == r_off.sample_indices
    @test isapprox(r_chain.samples, r_off.samples; atol = 1e-6)

    # :sorted solves out of draw order and records the mapping back
    @test sort(r_sorted.sample_indices) == r_off.sample_indices
    for j in eachindex(r_sorted.sample_indices)
        @test isapprox(
            r_sorted.samples[:, j],
            r_off.samples[:, r_sorted.sample_indices[j]];
            atol = 1e-6,
        )
    end
end

@testitem "warm start scheduling validates its inputs" tags = [:integration] setup =
    [MCSetup] begin
    prob = case5_problem()
    @test_throws ArgumentError solve(prob, MonteCarlo(n = 10, warmstart = :backwards))

    struct ColdBackend <: ProbabilisticPowerFlow.AbstractBackend end
    cold = PPFProblem(ColdBackend(), prob.model, prob.qois)
    @test_throws ArgumentError solve(cold, MonteCarlo(n = 10, warmstart = :chain))
end

@testitem "warm start chain recovers after divergence" tags = [:integration] setup =
    [MCSetup] begin
    # Near the loadability nose some samples diverge. The chain must restart cold
    # after each failure and keep the bookkeeping intact.
    prob = case5_problem(load_scale = 4.4, std_frac = 0.15)
    r = solve(prob, MonteCarlo(n = 400, warmstart = :chain); rng = Xoshiro(2026))
    @test !isempty(r.failures)
    @test n_converged(r) > 0
    @test n_converged(r) + length(r.failures) == r.n_samples
    @test r.n_solves == r.n_samples
end

@testitem "solve and solve! extend the CommonSolve interface" tags = [:unit, :fast] setup =
    [MCSetup] begin
    import CommonSolve
    import NonlinearSolve

    # the point of extending CommonSolve: loading this package next to a SciML
    # package gives one solve, not an ambiguous binding that errors on use
    @test solve === CommonSolve.solve
    @test solve! === CommonSolve.solve!
    @test parentmodule(solve) === CommonSolve

    prob = case5_problem()
    r = CommonSolve.solve(prob, MonteCarlo(n = 20); rng = Xoshiro(1))
    @test n_converged(r) == 20

    state = init_state(prob.backend, targets(prob.model))
    set_injections!(state, prob.backend, [0.45, 0.60, 0.30])
    @test CommonSolve.solve!(state, prob.backend).converged
end
