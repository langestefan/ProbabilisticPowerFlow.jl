@testsnippet PMCase5 begin
    using PowerModels
    PowerModels.silence()

    # MATPOWER mirror of src/case5.jl: bus 1 slack at 1.06, bus 2 PV at 1.04 with
    # 0.40 pu generation, buses 3-5 PQ, same seven branches with total charging.
    const CASE5_M = """
    function mpc = case5
    mpc.version = '2';
    mpc.baseMVA = 100.0;
    mpc.bus = [
        1  3   0   0  0  0  1  1.06  0  230  1  1.1  0.9;
        2  2  20  10  0  0  1  1.04  0  230  1  1.1  0.9;
        3  1  45  15  0  0  1  1.00  0  230  1  1.1  0.9;
        4  1  40   5  0  0  1  1.00  0  230  1  1.1  0.9;
        5  1  60  10  0  0  1  1.00  0  230  1  1.1  0.9;
    ];
    mpc.gen = [
        1   0  0  300  -300  1.06  100  1  250  -250  0  0  0  0  0  0  0  0  0  0  0;
        2  40  0  300  -300  1.04  100  1  250  -250  0  0  0  0  0  0  0  0  0  0  0;
    ];
    mpc.branch = [
        1  2  0.02  0.06  0.06  250  250  250  0  0  1  -360  360;
        1  3  0.08  0.24  0.05  250  250  250  0  0  1  -360  360;
        2  3  0.06  0.18  0.04  250  250  250  0  0  1  -360  360;
        2  4  0.06  0.18  0.04  250  250  250  0  0  1  -360  360;
        2  5  0.04  0.12  0.03  250  250  250  0  0  1  -360  360;
        3  4  0.01  0.03  0.02  250  250  250  0  0  1  -360  360;
        4  5  0.08  0.24  0.05  250  250  250  0  0  1  -360  360;
    ];
    """

    pm_case5() = PowerModels.parse_matpower(IOBuffer(CASE5_M))
    load_at(data, bus) = only(id for (id, l) in data["load"] if l["load_bus"] == bus)
end

@testitem "construction validates the data dict" tags = [:integration, :powermodels] setup =
    [PMCase5] begin
    data = pm_case5()
    @test PowerModelsBackend(data) isa AbstractBackend

    bad = deepcopy(data)
    bad["per_unit"] = false
    @test_throws ArgumentError PowerModelsBackend(bad)

    bad = deepcopy(data)
    bad["bus"]["2"]["bus_type"] = 3
    @test_throws ArgumentError PowerModelsBackend(bad)

    @test_throws ArgumentError PowerModelsBackend(Dict{String,Any}("per_unit" => true))
end

@testitem "contract errors on bad refs" tags = [:integration, :powermodels] setup =
    [PMCase5] begin
    data = pm_case5()
    backend = PowerModelsBackend(data)

    # unknown load id
    @test_throws ArgumentError init_state(backend, [ComponentRef(:load, "99", :pd)])
    # unsupported kind and field
    @test_throws ArgumentError init_state(backend, [ComponentRef(:bus, "3", :pd)])
    @test_throws ArgumentError init_state(backend, [ComponentRef(:load, "1", :pg)])
    # reactive load at the PV bus 2
    @test_throws ArgumentError init_state(
        backend,
        [ComponentRef(:load, load_at(data, 2), :qd)],
    )
    # inactive component
    inactive = deepcopy(data)
    id3 = load_at(inactive, 3)
    inactive["load"][id3]["status"] = 0
    @test_throws ArgumentError init_state(
        PowerModelsBackend(inactive),
        [ComponentRef(:load, id3, :pd)],
    )
end

@testitem "cross-backend agreement with the reference backend" tags =
    [:integration, :powermodels] setup = [PMCase5] begin
    data = pm_case5()
    pm = PowerModelsBackend(data)
    ref = ReferenceBackend(case5())

    ref_refs = [ComponentRef(:load, "3", :pd), ComponentRef(:load, "5", :pd)]
    pm_refs = [
        ComponentRef(:load, load_at(data, 3), :pd),
        ComponentRef(:load, load_at(data, 5), :pd),
    ]
    ref_state = init_state(ref, ref_refs)
    pm_state = init_state(pm, pm_refs)

    for x in ([0.45, 0.60], [0.55, 0.72], [0.30, 1.20])
        set_injections!(ref_state, ref, x)
        set_injections!(pm_state, pm, x)
        @test solve!(ref_state, ref).converged
        @test solve!(pm_state, pm).converged
        for bus = 1:5
            @test extract(ref_state, ref, VoltageMagnitude(bus)) ≈
                  extract(pm_state, pm, VoltageMagnitude(bus)) atol = 1e-6
            @test extract(ref_state, ref, VoltageAngle(bus)) ≈
                  extract(pm_state, pm, VoltageAngle(bus)) atol = 1e-6
        end
        @test extract(ref_state, ref, BranchActivePower(1, 3)) ≈
              extract(pm_state, pm, BranchActivePower(1, 3)) atol = 1e-6
        @test extract(ref_state, ref, BranchActivePower(3, 1)) ≈
              extract(pm_state, pm, BranchActivePower(3, 1)) atol = 1e-6
    end
end

@testitem "extract QoIs" tags = [:integration, :powermodels] setup = [PMCase5] begin
    data = pm_case5()
    backend = PowerModelsBackend(data)
    state = init_state(backend, [ComponentRef(:load, load_at(data, 5), :pd)])
    set_injections!(state, backend, [0.60])
    @test solve!(state, backend).converged

    @test 0.9 < extract(state, backend, VoltageMagnitude(5)) < 1.1
    @test extract(state, backend, VoltageAngle(5)) < 0.0
    @test extract(state, backend, VoltageAngle(1)) == 0.0

    # flow into the branch at both ends differs by the branch loss
    p13 = extract(state, backend, BranchActivePower(1, 3))
    p31 = extract(state, backend, BranchActivePower(3, 1))
    @test p13 > 0
    @test p13 + p31 > 0

    @test_throws ArgumentError extract(state, backend, BranchActivePower(1, 5))

    # ViolationEvent comes from the generic fallback
    tight = ViolationEvent(VoltageMagnitude(5), 0.99, 1.01)
    @test extract(state, backend, tight) in (0.0, 1.0)
    wide = ViolationEvent(VoltageMagnitude(5), 0.5, 1.5)
    @test extract(state, backend, wide) == 0.0
end

@testitem "warm start" tags = [:integration, :powermodels] setup = [PMCase5] begin
    data = pm_case5()
    backend = PowerModelsBackend(data)
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
end

@testitem "divergence is data, not an exception" tags = [:integration, :powermodels] setup =
    [PMCase5] begin
    data = pm_case5()
    backend = PowerModelsBackend(data)
    state = init_state(
        backend,
        [
            ComponentRef(:load, load_at(data, 3), :pd),
            ComponentRef(:load, load_at(data, 4), :pd),
            ComponentRef(:load, load_at(data, 5), :pd),
        ],
    )

    set_injections!(state, backend, [9.0, 8.0, 12.0])
    info = solve!(state, backend)
    @test !info.converged

    # the state stays usable: nominal loads solve again from a cold start
    set_injections!(state, backend, [0.45, 0.40, 0.60])
    again = solve!(state, backend)
    @test again.converged
    @test 0.9 < extract(state, backend, VoltageMagnitude(5)) < 1.1
end

@testitem "MC end-to-end through PPFProblem" tags = [:integration, :powermodels] setup =
    [PMCase5] begin
    using Distributions, Random, Statistics

    data = pm_case5()
    backend = PowerModelsBackend(data)

    model = UncertaintyModel(
        [GermVariable("p3", Normal(0.45, 0.02)), GermVariable("p5", Normal(0.60, 0.03))],
        [
            Assignment("p3", ComponentRef(:load, load_at(data, 3), :pd)),
            Assignment("p5", ComponentRef(:load, load_at(data, 5), :pd)),
        ],
        GaussianCopula([1.0 0.5; 0.5 1.0]),
    )
    qois = [VoltageMagnitude(5), ViolationEvent(VoltageMagnitude(5), 0.98, 1.10)]
    prob = PPFProblem(backend, model, qois)

    r = solve(prob, MonteCarlo(n = 500); rng = Xoshiro(2026))
    @test n_converged(r) == 500
    @test r.n_solves == 500

    # the MC mean matches a deterministic solve at the mean injections
    state = init_state(backend, targets(model))
    set_injections!(state, backend, [0.45, 0.60])
    @test solve!(state, backend).converged
    v5 = extract(state, backend, VoltageMagnitude(5))
    @test mean(r, VoltageMagnitude(5)) ≈ v5 atol = 0.005
    @test 0.0 <= violation_probability(r, qois[2]) <= 1.0
end
