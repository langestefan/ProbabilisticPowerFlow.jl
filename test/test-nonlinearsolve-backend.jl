@testitem "NonlinearSolve backend agrees with the PowerModels backend" tags =
    [:integration, :nonlinearsolve] setup = [PMCase5] begin
    import NonlinearSolve

    data = pm_case5()
    nls = PowerModelsBackend(data; solver = NonlinearSolve.NewtonRaphson())
    pm = PowerModelsBackend(data)
    refs = [
        ComponentRef(:load, load_at(data, 3), :pd),
        ComponentRef(:load, load_at(data, 5), :pd),
    ]
    s_nls = init_state(nls, refs)
    s_pm = init_state(pm, refs)

    # both backends solve the same equations, so the solutions agree to solver
    # tolerance at several load points
    for x in ([0.45, 0.60], [0.55, 0.72], [0.30, 1.20])
        set_injections!(s_nls, nls, x)
        set_injections!(s_pm, pm, x)
        @test solve!(s_nls, nls).converged
        @test solve!(s_pm, pm).converged
        for bus = 1:5
            @test extract(s_nls, nls, VoltageMagnitude(bus)) ≈
                  extract(s_pm, pm, VoltageMagnitude(bus)) atol = 1e-6
            @test extract(s_nls, nls, VoltageAngle(bus)) ≈
                  extract(s_pm, pm, VoltageAngle(bus)) atol = 1e-6
        end
        @test extract(s_nls, nls, BranchActivePower(1, 3)) ≈
              extract(s_pm, pm, BranchActivePower(1, 3)) atol = 1e-6
    end
end

@testitem "NonlinearSolve backend warm starts and records divergence" tags =
    [:integration, :nonlinearsolve] setup = [PMCase5] begin
    import NonlinearSolve

    data = pm_case5()
    backend = PowerModelsBackend(data; solver = NonlinearSolve.NewtonRaphson())
    @test supports_warmstart(backend)

    state = init_state(backend, [ComponentRef(:load, load_at(data, 5), :pd)])
    set_injections!(state, backend, [0.60])
    cold = solve!(state, backend)
    @test cold.converged

    set_injections!(state, backend, [0.61])
    warm = solve!(state, backend; warmstart = state)
    @test warm.converged
    @test warm.iterations <= cold.iterations
    @test_throws ArgumentError solve!(state, backend; warmstart = 1.0)

    # divergence is data, and the state recovers on the reused cache
    set_injections!(state, backend, [12.0])
    @test !solve!(state, backend).converged
    set_injections!(state, backend, [0.60])
    again = solve!(state, backend)
    @test again.converged
    @test 0.9 < extract(state, backend, VoltageMagnitude(5)) < 1.1
end

@testitem "NonlinearSolve backend reuses its cache without allocating" tags =
    [:integration, :nonlinearsolve] setup = [PMCase5] begin
    import NonlinearSolve

    data = pm_case5()
    backend = PowerModelsBackend(data; solver = NonlinearSolve.NewtonRaphson())
    state = init_state(backend, [ComponentRef(:load, load_at(data, 5), :pd)])
    set_injections!(state, backend, [0.60])
    solve!(state, backend)
    solve!(state, backend; warmstart = state)

    set_injections!(state, backend, [0.61])
    bytes = @allocated solve!(state, backend; warmstart = state)
    # the PowerModels NLsolve path allocates megabytes per solve on this case
    @test bytes < 200_000
end

@testitem "NonlinearSolve backend runs MC end-to-end with tasks" tags =
    [:integration, :nonlinearsolve] setup = [PMCase5] begin
    import NonlinearSolve
    using Distributions, Random, Statistics

    data = pm_case5()
    backend = PowerModelsBackend(data; solver = NonlinearSolve.NewtonRaphson())
    model = UncertaintyModel(
        [GermVariable("p3", Normal(0.45, 0.02)), GermVariable("p5", Normal(0.60, 0.03))],
        [
            Assignment("p3", ComponentRef(:load, load_at(data, 3), :pd)),
            Assignment("p5", ComponentRef(:load, load_at(data, 5), :pd)),
        ],
        GaussianCopula([1.0 0.5; 0.5 1.0]),
    )
    prob = PPFProblem(backend, model, AbstractQoI[VoltageMagnitude(5)])

    r1 = solve(prob, MonteCarlo(n = 200, warmstart = :chain); rng = Xoshiro(9))
    r4 = solve(prob, MonteCarlo(n = 200, warmstart = :chain); rng = Xoshiro(9), ntasks = 4)
    @test n_converged(r1) == 200
    @test r1.sample_indices == r4.sample_indices
    @test isapprox(r1.samples, r4.samples; atol = 1e-6)

    state = init_state(backend, ComponentRef[])
    solve!(state, backend)
    v5 = extract(state, backend, VoltageMagnitude(5))
    @test mean(r1, VoltageMagnitude(5)) ≈ v5 atol = 0.005
end

@testitem "solver choice is validated at construction" tags =
    [:integration, :nonlinearsolve] setup = [PMCase5] begin
    import NonlinearSolve

    data = pm_case5()
    @test PowerModelsBackend(data; solver = :nlsolve) isa AbstractBackend
    @test PowerModelsBackend(data; solver = NonlinearSolve.TrustRegion()) isa
          AbstractBackend
    @test_throws ArgumentError PowerModelsBackend(data; solver = :bogus)
    @test_throws ArgumentError PowerModelsBackend(data; solver = 42)
end
