@testsnippet MCCase5 begin
    using Distributions: Normal
    using Copulas: GaussianCopula
    using Random: Xoshiro
    using Statistics: mean, std

    const LOAD_BUSES = 3:5
    const VM5 = VoltageMagnitude(5)
    const BAND = ViolationEvent(VM5, 0.95, 1.011)
    const QOIS = AbstractQoI[VM5, VoltageAngle(5), BranchActivePower(2, 5), BAND]

    # correlated Normal loads at buses 3 to 5, with mean scale * base and std rel * mean
    function load_problem(data; scale = 1.0, rel = 0.1)
        vars = map(LOAD_BUSES) do bus
            mu = scale * base_pd(data, bus)
            return GermVariable("pd$(bus)", Normal(mu, rel * mu))
        end
        assigns = [
            Assignment("pd$(bus)", ComponentRef(ComponentField.Pd, load_at(data, bus)))
                for bus in LOAD_BUSES
        ]
        corr = [
            1.0 0.5 0.5
            0.5 1.0 0.5
            0.5 0.5 1.0
        ]
        model = UncertaintyModel(vars, assigns, GaussianCopula(corr))
        return PPFProblem(PowerModelsBackend(data), model, QOIS)
    end
end

@testitem "MonteCarlo rejects a non-positive sample count" tags = [:unit, :fast] begin
    @test MonteCarlo().n == 1000
    @test MonteCarlo(n = 5).n == 5
    @test_throws ArgumentError MonteCarlo(n = 0)
end

@testitem "MonteCarlo at nominal loading converges everywhere" tags =
    [:integration, :powermodels] setup = [PMCase5, MCCase5] begin
    data = pm_case5()
    prob = load_problem(data)
    r = solve(prob, MonteCarlo(n = 2000); rng = Xoshiro(1))

    @test r.method == MonteCarlo(n = 2000)
    @test r.qois == QOIS
    @test r.n_samples == 2000
    @test r.n_solves == 2000
    @test n_converged(r) + length(r.failures) == r.n_samples
    @test failure_rate(r) == 0
    @test r.sample_indices == 1:2000

    # the band cuts through the distribution of VM5
    p = violation_probability(r, BAND)
    @test 0 < p < 1
    @test p == mean(qoi_samples(r, BAND))
    @test std(r, VM5) > 0

    # the sample mean is close to a deterministic solve at the mean injections
    b = prob.backend
    state = init_state(b, targets(prob.model))
    set_injections!(state, b, [base_pd(data, bus) for bus in LOAD_BUSES])
    @test solve!(state, b).converged
    for q in QOIS[1:3]
        @test mean(r, q) ≈ extract(state, b, q) rtol = 0.01 atol = 1.0e-3
    end
end

@testitem "MonteCarlo is reproducible from a seed" tags = [:integration, :powermodels] setup =
    [PMCase5, MCCase5] begin
    prob = load_problem(pm_case5())

    r1 = solve(prob, MonteCarlo(n = 50); rng = Xoshiro(7))
    r2 = solve(prob, MonteCarlo(n = 50); rng = Xoshiro(7))
    r3 = solve(prob, MonteCarlo(n = 50); rng = Xoshiro(8))

    @test r1.samples == r2.samples
    @test r1.samples != r3.samples
end

@testitem "MonteCarlo records diverged samples and keeps going" tags =
    [:integration, :powermodels] setup = [PMCase5, MCCase5] begin
    prob = load_problem(pm_case5(); scale = 5.0, rel = 0.3)
    r = solve(prob, MonteCarlo(n = 200); rng = Xoshiro(1))

    @test 0 < failure_rate(r) < 1
    @test n_converged(r) + length(r.failures) == r.n_samples
    @test r.n_solves == 200
    @test size(r.samples) == (length(QOIS), n_converged(r))

    # failures and converged samples partition the sample indices
    failed = [f.index for f in r.failures]
    @test sort(vcat(failed, r.sample_indices)) == 1:200

    # a failure keeps the point and injections that reproduce it
    f = first(r.failures)
    @test !f.info.converged
    @test length(f.u) == germ_dim(prob.model)
    @test f.injections ≈ to_physical(prob.model, f.u)

    @test isfinite(mean(r, VM5))
end

@testitem "The sample loop checks the germ dimension" tags = [:integration, :powermodels] setup =
    [PMCase5, MCCase5] begin
    prob = load_problem(pm_case5())
    @test_throws DimensionMismatch ProbabilisticPowerFlow.solve_samples(
        prob,
        MonteCarlo(n = 1),
        rand(2, 5),
    )
end
