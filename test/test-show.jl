@testsnippet ShowSetup begin
    using Distributions: Normal
    using Copulas: GaussianCopula
    using LinearAlgebra: I

    tree(x) = sprint(show, MIME"text/plain"(), x)
    line(x) = sprint(show, x)

    function show_model(d = 2)
        vars = [GermVariable("l$(k)", Normal(1.0, 0.1)) for k in 1:d]
        assigns = [Assignment("l$(k)", ComponentRef(ComponentField.Pd, k)) for k in 1:d]
        corr = fill(0.3, d, d) + 0.7I
        return UncertaintyModel(vars, assigns, GaussianCopula(corr))
    end
end

@testitem "The tree display draws its branches" tags = [:unit, :fast] begin
    entries = ["a: 1", "b" => ["c: 2", "d" => ["e: 3"]], "f: 4"]
    text = sprint(ProbabilisticPowerFlow.show_tree, "Header", entries)

    @test text == """
        Header
        ├ a: 1
        ├ b
        │ ├ c: 2
        │ └ d
        │   └ e: 3
        └ f: 4"""
end

@testitem "Long lists are summarized" tags = [:unit, :fast] setup = [ShowSetup] begin
    limit = ProbabilisticPowerFlow.TREE_LIMIT
    @test ProbabilisticPowerFlow.listing(1:limit) == string.(1:limit)

    long = ProbabilisticPowerFlow.listing(1:(limit + 7))
    @test length(long) == limit + 1
    @test last(long) == "7 more"

    text = tree(show_model(20))
    @test occursin("germ variables: 20", text)
    @test occursin("15 more", text)
    @test count(==('\n'), text) < 20
end

@testitem "Small types print on one line" tags = [:unit, :fast] setup = [ShowSetup] begin
    ref = ComponentRef(ComponentField.Pd, 3)
    @test line(ref) == "ComponentRef(ComponentField.Pd, 3)"
    @test ProbabilisticPowerFlow.ref_label(ref) == "Pd[3]"

    @test line(SolveInfo(true, 3, 1.0e-10)) ==
        "SolveInfo(converged, 3 iterations, residual 1.0e-10)"
    @test line(SolveInfo(false, -1, Inf)) == "SolveInfo(diverged, -1 iterations, residual Inf)"

    @test line(ViolationEvent(VoltageMagnitude(5), 0.95, 1.05)) ==
        "ViolationEvent(VoltageMagnitude(5), 0.95, 1.05)"
    @test startswith(line(GermVariable("a", Normal())), "GermVariable(\"a\", ")

    @test line(Assignment("a", ref)) == "\"a\" → Pd[3]"
    @test line(Assignment("a", ref, AffineTransform(2.0, 1.0))) ==
        "\"a\" → Pd[3] via AffineTransform(2.0, 1.0)"

    @test line(MonteCarlo(n = 7)) == "MonteCarlo(n = 7)"
end

@testitem "A model displays as a tree" tags = [:unit, :fast] setup = [ShowSetup] begin
    model = show_model()
    @test line(model) == "UncertaintyModel(2 germ variables, 2 assignments)"

    text = tree(model)
    @test startswith(text, "UncertaintyModel\n├ germ variables: 2")
    @test occursin("│ ├ l1: ", text)
    @test occursin("μ=1.0, σ=0.1", text)
    @test occursin("│ ├ \"l1\" → Pd[1]", text)
    @test endswith(text, "└ dependence: GaussianCopula(d = 2)")
end

@testitem "A result displays its counts and failures" tags = [:unit, :fast] begin
    qois = AbstractQoI[VoltageMagnitude(2), VoltageAngle(2)]
    samples = [
        1.0 0.94 1.06
        0.1 0.2 0.3
    ]
    failure = FailedSample(3, [0.5], [2.0], SolveInfo(false, -1, Inf))
    r = PPFResult(MonteCarlo(n = 4), qois, samples, [1, 2, 4], [failure], 4, 4)

    @test sprint(show, r) == "PPFResult{MonteCarlo}(3/4 converged)"
    @test sprint(show, MIME"text/plain"(), r) == """
        PPFResult{MonteCarlo}
        ├ method: MonteCarlo(n = 4)
        ├ samples: 3 converged of 4, in 4 solves
        ├ failures: 1, 25.0% of samples
        └ quantities of interest: 2
          ├ VoltageMagnitude(2)
          └ VoltageAngle(2)"""

    clean = PPFResult(MonteCarlo(n = 3), qois, samples, [1, 2, 3], FailedSample[], 3, 3)
    @test occursin("├ failures: none", sprint(show, MIME"text/plain"(), clean))
end

@testitem "The PowerModels backend hides its data dictionary" tags =
    [:integration, :powermodels] setup = [ShowSetup, PMCase5] begin
    data = pm_case5()
    backend = PowerModelsBackend(data)
    ref = ComponentRef(ComponentField.Pd, load_at(data, 5))

    @test line(backend) == "PowerModelsBackend(5 buses)"
    text = tree(backend)
    @test startswith(text, "PowerModelsBackend\n├ network: case5\n│ ├ buses: 5")
    @test occursin("│ ├ branches: 7", text)
    @test endswith(text, "└ algorithm: NativeNewton")

    state = init_state(backend, [ref])
    @test line(state) == "PMState(5 buses, 1 injection slots, unsolved)"
    @test occursin("└ warm start available: false", tree(state))
    solve!(state, backend)
    @test line(state) == "PMState(5 buses, 1 injection slots, solved)"
    @test occursin("└ warm start available: true", tree(state))

    model = UncertaintyModel([GermVariable("l5", Normal(0.6, 0.06))], [Assignment("l5", ref)])
    prob = PPFProblem(backend, model, [VoltageMagnitude(5)])
    @test line(prob) == "PPFProblem(PowerModelsBackend(5 buses), 1 germ variables, 1 qois)"
    @test startswith(tree(prob), "PPFProblem\n├ backend: PowerModelsBackend(5 buses)")
    @test occursin("├ model: UncertaintyModel(1 germ variables, 1 assignments)", tree(prob))

    for x in (backend, state, prob)
        @test length(line(x)) < 100
        @test length(tree(x)) < 600
    end
end
