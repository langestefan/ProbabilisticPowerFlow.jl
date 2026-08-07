@testsnippet ShowSetup begin
    using Distributions, Random

    function show_model(d = 2)
        vars = [GermVariable("l$(k)", Normal(1.0, 0.1)) for k = 1:d]
        assigns = [Assignment("l$(k)", ComponentRef(:load, "$(k + 2)", :pd)) for k = 1:d]
        R = [i == j ? 1.0 : 0.3 for i = 1:d, j = 1:d]
        return UncertaintyModel(vars, assigns, GaussianCopula(R))
    end

    tree(x) = sprint(show, MIME"text/plain"(), x)
    line(x) = sprint(show, x)
end

@testitem "the tree display draws its branches" tags = [:unit, :fast] setup = [ShowSetup] begin
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

@testitem "long lists are summarized, not printed whole" tags = [:unit, :fast] setup =
    [ShowSetup] begin
    limit = ProbabilisticPowerFlow.TREE_LIMIT
    short = ProbabilisticPowerFlow.listing(1:limit)
    long = ProbabilisticPowerFlow.listing(1:(limit+7))

    @test short == string.(1:limit)
    @test length(long) == limit + 1
    @test last(long) == "7 more"

    text = tree(show_model(20))
    @test occursin("germ variables: 20", text)
    @test occursin("15 more", text)
    @test count(==('\n'), text) < 20
end

@testitem "models and problems display as a tree" tags = [:unit, :fast] setup = [ShowSetup] begin
    model = show_model()
    prob = PPFProblem(
        ReferenceBackend(case5()),
        model,
        [VoltageMagnitude(5), ViolationEvent(VoltageMagnitude(5), 0.95, 1.05)],
    )

    text = tree(prob)
    @test startswith(text, "PPFProblem\n├ backend: ReferenceBackend(5 buses)")
    @test occursin("│ ├ germ variables: 2", text)
    # the marginal prints through its own show, qualified or not depending on scope
    @test occursin("│ │ ├ l1: ", text)
    @test occursin("μ=1.0, σ=0.1", text)
    @test occursin("│ │ ├ \"l1\" → load[\"3\"].pd", text)
    @test occursin("│ └ dependence: GaussianCopula(d = 2)", text)
    @test occursin("└ quantities of interest: 2", text)
    @test occursin("  └ ViolationEvent(VoltageMagnitude(5), 0.95, 1.05)", text)

    @test line(prob) == "PPFProblem(ReferenceBackend(5 buses), 2 germ variables, 2 qois)"
    @test line(model) == "UncertaintyModel(2 germ variables, 2 assignments)"
end

@testitem "results and methods display as a tree" tags = [:unit, :fast] setup = [ShowSetup] begin
    prob = PPFProblem(ReferenceBackend(case5()), show_model(), [VoltageMagnitude(5)])
    r = solve(prob, MonteCarlo(n = 20, warmstart = :chain); rng = Xoshiro(1))
    text = tree(r)

    @test startswith(text, "PPFResult{MonteCarlo}\n├ method: MonteCarlo(n = 20)")
    @test occursin("│ ├ warmstart: :chain", text)
    @test occursin("├ samples: 20 converged of 20, in 20 solves", text)
    @test occursin("├ failures: none", text)
    @test occursin("├ inputs kept: no", text)
    @test line(r) == "PPFResult{MonteCarlo}(20/20 converged)"

    @test tree(MonteCarlo(n = 7)) ==
          "MonteCarlo\n├ n: 7\n├ failure_policy: :record\n├ warmstart: :off\n└ keep_inputs: false"
    @test line(LatinHypercube(n = 7)) == "LatinHypercube(n = 7)"
end

@testitem "a failing run reports its failure share" tags = [:integration] setup =
    [ShowSetup] begin
    using Statistics

    # past the loadability nose a fraction of the samples diverges
    base = 4.4 .* [0.45, 0.40, 0.60]
    vars = [
        GermVariable("l$(bus)", Normal(base[k], 0.15base[k])) for (k, bus) in enumerate(3:5)
    ]
    assigns = [Assignment("l$(bus)", ComponentRef(:load, "$(bus)", :pd)) for bus = 3:5]
    model = UncertaintyModel(vars, assigns, IndependentCopula())
    prob =
        PPFProblem(ReferenceBackend(case5(load_scale = 4.4)), model, [VoltageMagnitude(5)])
    r = solve(prob, MonteCarlo(n = 200); rng = Xoshiro(2026))

    @test !isempty(r.failures)
    @test occursin("% of the budget", tree(r))
    @test occursin("keep_inputs", tree(r))
end

@testitem "the reference backend keeps its display small" tags = [:unit, :fast] setup =
    [ShowSetup] begin
    backend = ReferenceBackend(case5())
    state = init_state(backend, [ComponentRef(:load, "3", :pd)])
    solve!(state, backend)

    @test line(backend) == "ReferenceBackend(5 buses)"
    @test occursin("├ network: NetworkData(5 buses, 7 branches)", tree(backend))
    @test occursin("├ buses: 5", tree(case5()))
    @test occursin("│ ├ slack: 1", tree(case5()))
    @test line(state) == "RefState(5 buses, 1 injection slots)"
    @test occursin("└ unknowns: ", tree(state))

    # the point of all this: no display carries the network arrays
    for x in (backend, case5(), state)
        @test length(line(x)) < 120
        @test length(tree(x)) < 300
    end
end
