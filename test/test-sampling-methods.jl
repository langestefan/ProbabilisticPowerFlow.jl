@testitem "LHS stratifies every margin" tags = [:unit, :fast] begin
    using Random

    n = 40
    U = ProbabilisticPowerFlow.lhs_points(Xoshiro(1), 5, n)
    @test size(U) == (5, n)
    @test all(0 .< U .< 1)
    for k = 1:5
        # exactly one point per stratum in every dimension
        @test sort(floor.(Int, U[k, :] .* n)) == collect(0:(n-1))
    end
end

@testitem "LHS end-to-end on case5" tags = [:integration] setup = [MCSetup] begin
    prob = case5_problem()
    r = solve(prob, LatinHypercube(n = 400); rng = Xoshiro(3))
    @test r.n_samples == 400
    @test r.n_solves == 400
    @test isempty(r.failures)

    state = init_state(prob.backend, ComponentRef[])
    solve!(state, prob.backend)
    v5_det = extract(state, prob.backend, VoltageMagnitude(5))
    @test mean(r, VoltageMagnitude(5)) ≈ v5_det atol = 0.005

    r2 = solve(prob, LatinHypercube(n = 400); rng = Xoshiro(3))
    @test r.samples == r2.samples
end

@testitem "LHS reduces the variance of the mean" tags = [:integration] setup =
    [MCSetup] begin
    # Repeated small-n estimates of the mean of vm5. The LHS estimates must
    # scatter less than the plain MC estimates. Deterministic under the seeds.
    prob = case5_problem()
    q = VoltageMagnitude(5)
    mc = [mean(solve(prob, MonteCarlo(n = 40); rng = Xoshiro(100 + s)), q) for s = 1:12]
    lhs =
        [mean(solve(prob, LatinHypercube(n = 40); rng = Xoshiro(100 + s)), q) for s = 1:12]
    @test std(lhs) < std(mc)
end

@testitem "Sobol end-to-end on case5" tags = [:integration] setup = [MCSetup] begin
    using Sobol

    prob = case5_problem()
    r = solve(prob, SobolSampling(n = 400); rng = Xoshiro(5))
    @test r.n_samples == 400
    @test r.n_solves == 400
    @test isempty(r.failures)

    state = init_state(prob.backend, ComponentRef[])
    solve!(state, prob.backend)
    v5_det = extract(state, prob.backend, VoltageMagnitude(5))
    @test mean(r, VoltageMagnitude(5)) ≈ v5_det atol = 0.005

    # the Cranley-Patterson shift ties the estimate to the seed
    r2 = solve(prob, SobolSampling(n = 400); rng = Xoshiro(5))
    @test r.samples == r2.samples
    r3 = solve(prob, SobolSampling(n = 400); rng = Xoshiro(6))
    @test r.samples != r3.samples
end

@testitem "Sobol converges faster than MC on case5" tags = [:integration] setup =
    [MCSetup] begin
    using Sobol

    prob = case5_problem()
    q = VoltageMagnitude(5)
    mc = [mean(solve(prob, MonteCarlo(n = 64); rng = Xoshiro(200 + s)), q) for s = 1:12]
    qmc =
        [mean(solve(prob, SobolSampling(n = 64); rng = Xoshiro(200 + s)), q) for s = 1:12]
    @test std(qmc) < std(mc)
end

@testitem "sampling methods share the warm start modes" tags = [:integration] setup =
    [MCSetup] begin
    using Sobol

    prob = case5_problem()
    for method in (LatinHypercube, SobolSampling)
        r_off = solve(prob, method(n = 100, warmstart = :off); rng = Xoshiro(9))
        r_chain = solve(prob, method(n = 100, warmstart = :chain); rng = Xoshiro(9))
        r_sorted = solve(prob, method(n = 100, warmstart = :sorted); rng = Xoshiro(9))
        @test isapprox(r_chain.samples, r_off.samples; atol = 1e-6)
        @test sort(r_sorted.sample_indices) == r_off.sample_indices
        for j in eachindex(r_sorted.sample_indices)
            @test isapprox(
                r_sorted.samples[:, j],
                r_off.samples[:, r_sorted.sample_indices[j]];
                atol = 1e-6,
            )
        end
        @test_throws ArgumentError solve(prob, method(n = 10, warmstart = :backwards))
    end
end

@testitem "QMCSampling adapts QuasiMonteCarlo samplers" tags = [:integration] setup =
    [MCSetup] begin
    using QuasiMonteCarlo

    prob = case5_problem()
    state = init_state(prob.backend, ComponentRef[])
    solve!(state, prob.backend)
    v5_det = extract(state, prob.backend, VoltageMagnitude(5))

    for sampler in (SobolSample(), HaltonSample())
        r = solve(prob, QMCSampling(sampler; n = 400))
        @test r.n_samples == 400
        @test r.n_solves == 400
        @test isempty(r.failures)
        @test mean(r, VoltageMagnitude(5)) ≈ v5_det atol = 0.005
    end
end

@testitem "QMCSampling randomization is owned by the sampler" tags = [:integration] setup =
    [MCSetup] begin
    using QuasiMonteCarlo

    prob = case5_problem()
    scrambled(seed) =
        SobolSample(R = OwenScramble(base = 2, pad = 32, rng = Xoshiro(seed)))
    r1 = solve(prob, QMCSampling(scrambled(1); n = 256))
    r2 = solve(prob, QMCSampling(scrambled(1); n = 256))
    r3 = solve(prob, QMCSampling(scrambled(2); n = 256))
    @test r1.samples == r2.samples
    @test r1.samples != r3.samples

    # the solve rng does not touch the points
    r4 = solve(prob, QMCSampling(scrambled(1); n = 256); rng = Xoshiro(99))
    @test r1.samples == r4.samples
end

@testitem "QMCSampling shares the warm start modes" tags = [:integration] setup =
    [MCSetup] begin
    using QuasiMonteCarlo

    prob = case5_problem()
    mk(ws) = QMCSampling(SobolSample(); n = 100, warmstart = ws)
    r_off = solve(prob, mk(:off))
    r_chain = solve(prob, mk(:chain))
    r_sorted = solve(prob, mk(:sorted))
    @test isapprox(r_chain.samples, r_off.samples; atol = 1e-6)
    @test sort(r_sorted.sample_indices) == r_off.sample_indices
    @test_throws ArgumentError solve(prob, mk(:backwards))
end

@testitem "keep_inputs stores the converged u points" tags = [:integration] setup =
    [MCSetup] begin
    prob = case5_problem()
    r = solve(prob, MonteCarlo(n = 50, keep_inputs = true); rng = Xoshiro(21))
    @test r.u isa Matrix{Float64}
    @test size(r.u) == (3, n_converged(r))
    @test all(0 .< r.u .< 1)

    # a stored u point reproduces its recorded QoI through a fresh solve
    x = to_physical(prob.model, r.u[:, 7])
    state = init_state(prob.backend, targets(prob.model))
    set_injections!(state, prob.backend, x)
    @test solve!(state, prob.backend).converged
    @test extract(state, prob.backend, VoltageMagnitude(5)) ≈ r.samples[1, 7] atol = 1e-10

    # off by default, and off through the shared matrix path too
    @test solve(prob, MonteCarlo(n = 20); rng = Xoshiro(1)).u === nothing
    rs = solve(prob, LatinHypercube(n = 20, keep_inputs = true); rng = Xoshiro(1))
    @test size(rs.u) == (3, n_converged(rs))

    # failed samples keep their u in failures, converged u stays aligned
    heavy = case5_problem(load_scale = 4.4, std_frac = 0.15)
    rh = solve(heavy, MonteCarlo(n = 100, keep_inputs = true); rng = Xoshiro(2026))
    @test !isempty(rh.failures)
    @test size(rh.u, 2) == n_converged(rh)
end
