@testsnippet PMCase5 begin
    using PowerModels
    PowerModels.silence()

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

    load_at(data, bus) =
        parse(Int, only(id for (id, l) in data["load"] if l["load_bus"] == bus))
    gen_at(data, bus) =
        parse(Int, only(id for (id, g) in data["gen"] if g["gen_bus"] == bus))
    base_pd(data, bus) = data["load"][string(load_at(data, bus))]["pd"]
end

@testitem "Construction validates the network data" tags = [:integration, :powermodels] setup =
    [PMCase5] begin
    data = pm_case5()
    b = PowerModelsBackend(data)
    @test b isa AbstractPFBackend

    @test b isa PowerModelsBackend
    @test parentmodule(typeof(b)) === ProbabilisticPowerFlow
    @test b.alg isa PowerModels.NativeNewton

    @test_throws ArgumentError PowerModelsBackend(Dict{String,Any}("per_unit" => true))

    notpu = deepcopy(data)
    notpu["per_unit"] = false
    @test_throws ArgumentError PowerModelsBackend(notpu)

    multi = deepcopy(data)
    multi["multinetwork"] = true
    @test_throws ArgumentError PowerModelsBackend(multi)

    twoslack = deepcopy(data)
    twoslack["bus"]["2"]["bus_type"] = 3
    @test PowerModelsBackend(twoslack) isa AbstractPFBackend

    noslack = deepcopy(data)
    noslack["bus"]["1"]["bus_type"] = 2
    @test_throws ArgumentError PowerModelsBackend(noslack)

    nogen = deepcopy(data)
    nogen["bus"]["3"]["bus_type"] = 2
    @test_throws ArgumentError PowerModelsBackend(nogen)
end

@testitem "The backend is not mutated by its caller" tags = [:integration, :powermodels] setup =
    [PMCase5] begin
    data = pm_case5()
    b = PowerModelsBackend(data)
    original = b.data["load"][string(load_at(data, 3))]["pd"]

    data["load"][string(load_at(data, 3))]["pd"] = 99.0
    @test b.data["load"][string(load_at(data, 3))]["pd"] == original
end

@testitem "init_state rejects references it cannot honour" tags =
    [:integration, :powermodels] setup = [PMCase5] begin
    data = pm_case5()
    b = PowerModelsBackend(data)

    ok = [ComponentRef(ComponentField.Pd, load_at(data, 3))]
    @test init_state(b, ok) !== nothing

    for field in (ComponentField.Vm, ComponentField.Vg, ComponentField.Qg)
        @test_throws ArgumentError init_state(b, [ComponentRef(field, 1)])
    end

    @test_throws ArgumentError init_state(b, [ComponentRef(ComponentField.Pd, 999)])

    @test_throws ArgumentError init_state(
        b,
        [ComponentRef(ComponentField.Pg, gen_at(data, 1))],
    )

    @test_throws ArgumentError init_state(
        b,
        [ComponentRef(ComponentField.Qd, load_at(data, 2))],
    )

    off = deepcopy(data)
    off["load"][string(load_at(data, 3))]["status"] = 0
    boff = PowerModelsBackend(off)
    @test_throws ArgumentError init_state(
        boff,
        [ComponentRef(ComponentField.Pd, load_at(data, 3))],
    )

    @test init_state(b, [ComponentRef(ComponentField.Pg, gen_at(data, 2))]) !== nothing
end

@testitem "A solve at base injections matches PowerModels itself" tags =
    [:integration, :powermodels] setup = [PMCase5] begin
    data = pm_case5()
    b = PowerModelsBackend(data)

    refs = [ComponentRef(ComponentField.Pd, load_at(data, bus)) for bus = 3:5]
    state = init_state(b, refs)
    set_injections!(state, b, [base_pd(data, bus) for bus = 3:5])
    info = solve!(state, b)

    @test info.converged
    @test info.iterations > 0
    @test isfinite(info.residual)

    ref = PowerModels.compute_ac_pf(data)["solution"]["bus"]
    for bus = 1:5
        @test extract(state, b, VoltageMagnitude(bus)) ≈ ref["$(bus)"]["vm"] atol = 1e-6
        @test extract(state, b, VoltageAngle(bus)) ≈ ref["$(bus)"]["va"] atol = 1e-6
    end
end

@testitem "Injections enter the parameter vector with the right sign" tags =
    [:integration, :powermodels] setup = [PMCase5] begin
    data = pm_case5()
    b = PowerModelsBackend(data)

    refs = [ComponentRef(ComponentField.Pd, load_at(data, 5))]
    state = init_state(b, refs)

    set_injections!(state, b, [base_pd(data, 5)])
    solve!(state, b)
    v_base = extract(state, b, VoltageMagnitude(5))

    set_injections!(state, b, [base_pd(data, 5) * 2])
    solve!(state, b)
    @test extract(state, b, VoltageMagnitude(5)) < v_base

    gstate = init_state(b, [ComponentRef(ComponentField.Pg, gen_at(data, 2))])
    set_injections!(gstate, b, [0.4])
    solve!(gstate, b)
    a_base = extract(gstate, b, VoltageAngle(2))

    set_injections!(gstate, b, [1.2])
    solve!(gstate, b)
    @test extract(gstate, b, VoltageAngle(2)) > a_base
end

@testitem "Two components on one bus accumulate" tags = [:integration, :powermodels] setup =
    [PMCase5] begin
    data = pm_case5()

    first_id = load_at(data, 4)
    second = deepcopy(data["load"][string(first_id)])
    second["index"] = 99
    second["pd"] = 0.1
    second["qd"] = 0.0
    data["load"]["99"] = second

    b = PowerModelsBackend(data)
    refs = [ComponentRef(ComponentField.Pd, first_id), ComponentRef(ComponentField.Pd, 99)]

    a = init_state(b, refs)
    set_injections!(a, b, [0.2, 0.3])
    @test solve!(a, b).converged

    c = init_state(b, refs)
    set_injections!(c, b, [0.4, 0.1])
    @test solve!(c, b).converged

    for bus = 1:5
        @test extract(a, b, VoltageMagnitude(bus)) ≈ extract(c, b, VoltageMagnitude(bus)) atol =
            1e-10
    end

    d = init_state(b, refs)
    set_injections!(d, b, [0.2, 0.2])
    solve!(d, b)
    @test extract(d, b, VoltageMagnitude(4)) != extract(a, b, VoltageMagnitude(4))
end

@testitem "An assignment is absolute, not a delta on the base value" tags =
    [:integration, :powermodels] setup = [PMCase5] begin
    data = pm_case5()
    b = PowerModelsBackend(data)
    refs = [ComponentRef(ComponentField.Pd, load_at(data, bus)) for bus = 3:5]

    state = init_state(b, refs)
    set_injections!(state, b, [base_pd(data, bus) for bus = 3:5])
    @test solve!(state, b).converged

    untouched = init_state(b, ComponentRef[])
    @test solve!(untouched, b).converged

    for bus = 1:5
        @test extract(state, b, VoltageMagnitude(bus)) ≈
              extract(untouched, b, VoltageMagnitude(bus)) atol = 1e-10
    end
end

@testitem "A solve depends only on its own injections" tags = [:integration, :powermodels] setup =
    [PMCase5] begin
    data = pm_case5()
    b = PowerModelsBackend(data)
    refs = [ComponentRef(ComponentField.Pd, load_at(data, bus)) for bus = 3:5]
    target = [0.5, 0.4, 0.7]

    fresh = init_state(b, refs)
    set_injections!(fresh, b, target)
    clean = solve!(fresh, b)

    used = init_state(b, refs)
    set_injections!(used, b, [1.5, 1.4, 1.7])
    solve!(used, b)
    set_injections!(used, b, target)
    after = solve!(used, b)

    @test after.iterations == clean.iterations
    @test after.residual ≈ clean.residual atol = 1e-14
    for bus = 1:5
        @test extract(used, b, VoltageMagnitude(bus)) ≈
              extract(fresh, b, VoltageMagnitude(bus)) atol = 1e-12
    end
end

@testitem "States from separate init_state calls are independent" tags =
    [:integration, :powermodels] setup = [PMCase5] begin
    data = pm_case5()
    b = PowerModelsBackend(data)
    refs = [ComponentRef(ComponentField.Pd, load_at(data, 5))]

    a = init_state(b, refs)
    c = init_state(b, refs)

    set_injections!(a, b, [0.6])
    solve!(a, b)
    set_injections!(c, b, [1.4])
    solve!(c, b)

    @test extract(a, b, VoltageMagnitude(5)) != extract(c, b, VoltageMagnitude(5))
    va = extract(a, b, VoltageMagnitude(5))
    solve!(c, b)
    @test extract(a, b, VoltageMagnitude(5)) == va
end

@testitem "A warm start converges in fewer iterations" tags = [:integration, :powermodels] setup =
    [PMCase5] begin
    data = pm_case5()
    b = PowerModelsBackend(data)
    @test supports_warmstart(b)

    refs = [ComponentRef(ComponentField.Pd, load_at(data, bus)) for bus = 3:5]

    seed = init_state(b, refs)
    set_injections!(seed, b, [0.46, 0.41, 0.61])
    @test solve!(seed, b).converged

    nearby = [0.45, 0.40, 0.60]
    cold = init_state(b, refs)
    set_injections!(cold, b, nearby)
    cold_info = solve!(cold, b)

    warm = init_state(b, refs)
    set_injections!(warm, b, nearby)
    warm_info = solve!(warm, b; warmstart = seed)

    @test warm_info.converged
    @test warm_info.iterations < cold_info.iterations
    @test extract(warm, b, VoltageMagnitude(5)) ≈ extract(cold, b, VoltageMagnitude(5)) atol =
        1e-8

    @test_throws ArgumentError solve!(warm, b; warmstart = :not_a_state)
end

@testitem "Divergence is returned, not thrown" tags = [:integration, :powermodels] setup =
    [PMCase5] begin
    data = pm_case5()
    b = PowerModelsBackend(data)
    refs = [ComponentRef(ComponentField.Pd, load_at(data, bus)) for bus = 3:5]
    state = init_state(b, refs)

    set_injections!(state, b, [200.0, 200.0, 200.0])
    bad = solve!(state, b)
    @test !bad.converged

    set_injections!(state, b, [0.45, 0.40, 0.60])
    good = solve!(state, b)
    @test good.converged
    @test 0.9 < extract(state, b, VoltageMagnitude(5)) < 1.1
end

@testitem "Branch flows are read from the solved state" tags = [:integration, :powermodels] setup =
    [PMCase5] begin
    data = pm_case5()
    b = PowerModelsBackend(data)
    state = init_state(b, ComponentRef[])
    @test solve!(state, b).converged

    pf = extract(state, b, BranchActivePower(1, 2))
    pt = extract(state, b, BranchActivePower(2, 1))
    @test pf != 0.0
    @test pf + pt > 0.0

    qf = extract(state, b, BranchReactivePower(1, 2))
    @test isfinite(qf)

    @test_throws ArgumentError extract(state, b, BranchActivePower(1, 4))
    @test_throws ArgumentError extract(state, b, VoltageMagnitude(99))
end

@testitem "Parallel branches make a branch flow ambiguous" tags =
    [:integration, :powermodels] setup = [PMCase5] begin
    data = pm_case5()
    data["branch"]["8"] = deepcopy(data["branch"]["1"])
    data["branch"]["8"]["index"] = 8
    b = PowerModelsBackend(data)

    state = init_state(b, ComponentRef[])
    @test solve!(state, b).converged
    @test_throws ArgumentError extract(state, b, BranchActivePower(1, 2))

    @test isfinite(extract(state, b, BranchActivePower(2, 4)))
end

@testitem "A ViolationEvent needs no backend method" tags = [:integration, :powermodels] setup =
    [PMCase5] begin
    data = pm_case5()
    b = PowerModelsBackend(data)
    state = init_state(b, ComponentRef[])
    solve!(state, b)

    vm = extract(state, b, VoltageMagnitude(5))
    @test extract(state, b, ViolationEvent(VoltageMagnitude(5), vm - 0.01, vm + 0.01)) ==
          0.0
    @test extract(state, b, ViolationEvent(VoltageMagnitude(5), vm + 0.01, vm + 0.02)) ==
          1.0
end

@testitem "set_injections! checks the length of its input" tags =
    [:integration, :powermodels] setup = [PMCase5] begin
    data = pm_case5()
    b = PowerModelsBackend(data)
    state = init_state(b, [ComponentRef(ComponentField.Pd, load_at(data, 3))])

    @test_throws DimensionMismatch set_injections!(state, b, [0.1, 0.2])
    @test_throws DimensionMismatch set_injections!(state, b, Float64[])
end
