@testsnippet PM begin
    using Distributions
    using Copulas
    using Random: Xoshiro
    import PowerModels
    using ProbabilisticPowerFlow: AffineTransform

    PowerModels.silence()

    CASE14 = joinpath(pkgdir(PowerModels), "test", "data", "matpower", "case14.m")
    case14() = PowerModels.parse_file(CASE14)

    # case14 bus types: 1 is the reference, 2/3/6/8 are PV, the rest PQ.
    # load 3 sits at PQ bus 4, load 1 at PV bus 2, gen 2 at PV bus 2, gen 1 at the
    # reference bus.
    PD4 = ComponentRef(ComponentField.Pd, 3)
    QD4 = ComponentRef(ComponentField.Qd, 3)
    PG2 = ComponentRef(ComponentField.Pg, 2)

    # the reference solution PowerModels itself produces from the untouched network
    function reference_voltages()
        res = PowerModels.compute_ac_pf(case14())
        @assert res["termination_status"]
        return res["solution"]["bus"]
    end
end

@testitem "At the base point the backend reproduces PowerModels" setup = [PM] tags = [:unit] begin
    data = case14()
    b = PowerModelsBackend(data)
    s = init_state(b, [PD4, QD4])

    # the base injections, so nothing is perturbed away from the reference case
    set_injections!(s, b, [data["load"]["3"]["pd"], data["load"]["3"]["qd"]])
    info = solve!(s, b)

    @test info.converged
    @test info.iterations > 0
    @test info.residual <= 1e-8

    ref = reference_voltages()
    for (i, bus) in ref
        n = parse(Int, i)
        @test extract(s, b, VoltageMagnitude(n)) ≈ bus["vm"] atol = 1e-7
        @test extract(s, b, VoltageAngle(n)) ≈ bus["va"] atol = 1e-7
    end
end

@testitem "Solving without set_injections! is the base case too" setup = [PM] tags = [:unit] begin
    b = PowerModelsBackend(case14())
    s = init_state(b, [PD4, QD4])

    # p0 comes out of instantiate_pf_data already holding the network's own injections
    @test solve!(s, b).converged
    @test extract(s, b, VoltageMagnitude(5)) ≈ reference_voltages()["5"]["vm"] atol = 1e-7
end

@testitem "Injections move the solution the right way" setup = [PM] tags = [:unit] begin
    data = case14()
    b = PowerModelsBackend(data)
    s = init_state(b, [PD4, QD4])
    pd, qd = data["load"]["3"]["pd"], data["load"]["3"]["qd"]

    set_injections!(s, b, [pd, qd])
    solve!(s, b)
    base = extract(s, b, VoltageMagnitude(4))

    # more load at bus 4 pulls its voltage down
    set_injections!(s, b, [pd + 0.5, qd])
    solve!(s, b)
    @test extract(s, b, VoltageMagnitude(4)) < base

    # a negative load is an injection, and pushes it back up
    set_injections!(s, b, [pd - 0.5, qd])
    solve!(s, b)
    @test extract(s, b, VoltageMagnitude(4)) > base

    # returning to the base injections returns to the base solution, so no state
    # leaks from one solve into the next
    set_injections!(s, b, [pd, qd])
    solve!(s, b)
    @test extract(s, b, VoltageMagnitude(4)) ≈ base atol = 1e-9
end

@testitem "Two assignments on one bus compose" setup = [PM] tags = [:unit] begin
    data = case14()
    b = PowerModelsBackend(data)
    pd = data["load"]["3"]["pd"]

    # bus 4 carries load 3; a second Pd assignment on the same load has to add, not
    # overwrite, because set_injections! resets to the base before accumulating
    one = init_state(b, [PD4])
    set_injections!(one, b, [pd + 0.3])
    solve!(one, b)

    two = init_state(b, [PD4, PD4])
    set_injections!(two, b, [pd + 0.1, pd + 0.2])
    solve!(two, b)

    @test extract(two, b, VoltageMagnitude(4)) ≈ extract(one, b, VoltageMagnitude(4)) atol =
        1e-9
end

@testitem "Generator setpoints are injections too" setup = [PM] tags = [:unit] begin
    data = case14()
    b = PowerModelsBackend(data)
    s = init_state(b, [PG2])
    pg = data["gen"]["2"]["pg"]

    set_injections!(s, b, [pg])
    solve!(s, b)
    base = extract(s, b, VoltageAngle(4))

    # generation enters p with the opposite sign to load, so raising pg at bus 2
    # must move bus 4's angle the other way from raising pd there
    set_injections!(s, b, [pg + 0.4])
    solve!(s, b)
    @test extract(s, b, VoltageAngle(4)) != base
end

@testitem "Warm and cold solves agree" setup = [PM] tags = [:unit] begin
    data = case14()
    b = PowerModelsBackend(data)
    @test supports_warmstart(b)
    pd, qd = data["load"]["3"]["pd"], data["load"]["3"]["qd"]

    prev = init_state(b, [PD4, QD4])
    set_injections!(prev, b, [pd, qd])
    @test solve!(prev, b).converged

    cold = init_state(b, [PD4, QD4])
    set_injections!(cold, b, [pd + 0.2, qd])
    cold_info = solve!(cold, b)

    warm = init_state(b, [PD4, QD4])
    set_injections!(warm, b, [pd + 0.2, qd])
    warm_info = solve!(warm, b; warmstart = prev)

    @test warm_info.converged
    @test extract(warm, b, VoltageMagnitude(4)) ≈ extract(cold, b, VoltageMagnitude(4)) atol =
        1e-8
    # a nearby starting point is the whole point of warm starting
    @test warm_info.iterations <= cold_info.iterations

    @test_throws ArgumentError solve!(warm, b; warmstart = :previous)
end

@testitem "States are independent" setup = [PM] tags = [:unit] begin
    data = case14()
    b = PowerModelsBackend(data)
    pd, qd = data["load"]["3"]["pd"], data["load"]["3"]["qd"]

    a = init_state(b, [PD4, QD4])
    c = init_state(b, [PD4, QD4])

    set_injections!(a, b, [pd + 1.0, qd])
    solve!(a, b)
    set_injections!(c, b, [pd, qd])
    solve!(c, b)

    # solving c must not have disturbed a's answer, and vice versa
    va = extract(a, b, VoltageMagnitude(4))
    solve!(a, b)
    @test extract(a, b, VoltageMagnitude(4)) ≈ va atol = 1e-12
    @test extract(c, b, VoltageMagnitude(4)) != va

    # the backend itself is never touched
    @test b.data["load"]["3"]["pd"] == data["load"]["3"]["pd"]
end

@testitem "Branch flows match calc_branch_flow_ac" setup = [PM] tags = [:unit] begin
    data = case14()
    b = PowerModelsBackend(data)
    s = init_state(b, [PD4])
    set_injections!(s, b, [data["load"]["3"]["pd"]])
    solve!(s, b)

    res = PowerModels.compute_ac_pf(data)
    PowerModels.update_data!(data, res["solution"])
    flows = PowerModels.calc_branch_flow_ac(data)
    br = first(values(data["branch"]))
    f, t = br["f_bus"], br["t_bus"]
    ref = flows["branch"]["$(br["index"])"]

    @test extract(s, b, BranchActivePower(f, t)) ≈ ref["pf"] atol = 1e-7
    @test extract(s, b, BranchReactivePower(f, t)) ≈ ref["qf"] atol = 1e-7
    @test extract(s, b, BranchActivePower(t, f)) ≈ ref["pt"] atol = 1e-7
    @test extract(s, b, BranchReactivePower(t, f)) ≈ ref["qt"] atol = 1e-7

    @test_throws ArgumentError extract(s, b, BranchActivePower(4, 14))
    @test_throws ArgumentError extract(s, b, VoltageMagnitude(99))
end

@testitem "init_state rejects refs that would be silently ignored" setup = [PM] tags =
    [:unit] begin
    b = PowerModelsBackend(case14())

    # gen 1 is at the reference bus, whose injection is an outcome of the solve
    @test_throws ArgumentError init_state(b, [ComponentRef(ComponentField.Pg, 1)])
    # load 1 is at PV bus 2, whose voltage setpoint absorbs reactive power
    @test_throws ArgumentError init_state(b, [ComponentRef(ComponentField.Qd, 1)])
    # but its active load is a genuine injection
    @test init_state(b, [ComponentRef(ComponentField.Pd, 1)]) isa Any
    # no such component
    @test_throws ArgumentError init_state(b, [ComponentRef(ComponentField.Pd, 999)])
    # fields that are setpoints or solve outcomes, not injections
    @test_throws ArgumentError init_state(b, [ComponentRef(ComponentField.Qg, 2)])
    @test_throws ArgumentError init_state(b, [ComponentRef(ComponentField.Vg, 2)])
    @test_throws ArgumentError init_state(b, [ComponentRef(ComponentField.Vm, 2)])
end

@testitem "An inactive component is a rejection, not a no-op" setup = [PM] tags = [:unit] begin
    data = case14()
    data["load"]["7"]["status"] = 0
    b = PowerModelsBackend(data)

    @test_throws ArgumentError init_state(b, [ComponentRef(ComponentField.Pd, 7)])
end

@testitem "The constructor rejects networks the power flow cannot represent" setup = [PM] tags =
    [:unit] begin
    @test PowerModelsBackend(case14()) isa Any

    # a generator at a PQ bus is what instantiate_pf_data asserts on
    d = case14()
    d["bus"]["6"]["bus_type"] = 1
    @test_throws ArgumentError PowerModelsBackend(d)

    # a PV bus with no generator behind it
    d = case14()
    d["gen"]["4"]["gen_status"] = 0
    @test_throws ArgumentError PowerModelsBackend(d)

    # no reference bus at all
    d = case14()
    d["bus"]["1"]["bus_type"] = 2
    @test_throws ArgumentError PowerModelsBackend(d)

    d = case14()
    delete!(d, "branch")
    @test_throws ArgumentError PowerModelsBackend(d)

    d = case14()
    d["per_unit"] = false
    @test_throws ArgumentError PowerModelsBackend(d)
end

@testitem "A diverged solve is data, not an exception" setup = [PM] tags = [:unit] begin
    data = case14()
    b = PowerModelsBackend(data)
    s = init_state(b, [PD4, QD4])
    pd, qd = data["load"]["3"]["pd"], data["load"]["3"]["qd"]

    # far past the loadability limit of the case: no solution exists to find
    set_injections!(s, b, [50.0, 20.0])
    info = solve!(s, b)
    @test info isa SolveInfo
    @test !info.converged

    # the state is still usable, so a diverged sample never poisons the next one
    set_injections!(s, b, [pd, qd])
    @test solve!(s, b).converged

    # running out of iterations is reported the same way, as data
    tight = PowerModelsBackend(data; maxiter = 1)
    t = init_state(tight, [PD4, QD4])
    set_injections!(t, tight, [pd, qd])
    @test !solve!(t, tight).converged

    # the bus state vectors now hold the final iterate, not a solution, so reading
    # one would be a plausible number that is simply wrong
    @test_throws ArgumentError extract(t, tight, VoltageMagnitude(4))
    @test_throws ArgumentError extract(t, tight, BranchActivePower(4, 5))
    # and an unsolved state has nothing to read at all
    @test_throws ArgumentError extract(init_state(b, [PD4]), b, VoltageMagnitude(4))
end

@testitem "A full Monte Carlo run on case14" setup = [PM] tags = [:integration] begin
    data = case14()
    b = PowerModelsBackend(data)

    variables =
        [GermVariable("load", Normal(1.0, 0.10)), GermVariable("wind", Weibull(2.0, 8.0))]
    assignments = [
        Assignment("load", PD4, AffineTransform(0.478, 0.0)),
        Assignment("load", QD4, AffineTransform(-0.039, 0.0)),
        # wind at PQ bus 5 as a negative load, 0.05 pu per m/s
        Assignment(
            "wind",
            ComponentRef(ComponentField.Pd, 4),
            AffineTransform(-0.05, 0.076),
        ),
    ]
    model = UncertaintyModel(variables, assignments, GaussianCopula([1.0 0.4; 0.4 1.0]))

    qois = [VoltageMagnitude(4), VoltageMagnitude(5), BranchActivePower(4, 5)]
    prob = PPFProblem(b, model, qois)
    r = solve(prob, MonteCarlo(n = 60, keep_inputs = true); rng = Xoshiro(20260902))

    @test n_converged(r) == 60
    @test all(0.9 .< qoi_samples(r, VoltageMagnitude(4)) .< 1.1)
    # the two neighbouring buses move together
    @test cor(qoi_samples(r, VoltageMagnitude(4)), qoi_samples(r, VoltageMagnitude(5))) >
          0.9

    # replaying a stored u point through the backend reproduces its recorded sample
    s = init_state(b, targets(model))
    set_injections!(s, b, to_physical(model, r.u[:, 7]))
    @test solve!(s, b).converged
    @test extract(s, b, VoltageMagnitude(4)) ≈ r.samples[1, 7] atol = 1e-9
end

@testitem "Concurrent sampling gives the same answer as serial" setup = [PM] tags =
    [:integration] begin
    b = PowerModelsBackend(case14())
    model = UncertaintyModel(
        [GermVariable("load", Normal(1.0, 0.15))],
        [
            Assignment("load", PD4, AffineTransform(0.478, 0.0)),
            Assignment("load", QD4, AffineTransform(-0.039, 0.0)),
        ],
    )
    prob = PPFProblem(b, model, [VoltageMagnitude(4)])

    serial = solve(prob, MonteCarlo(n = 40); rng = Xoshiro(1))
    par = solve(prob, MonteCarlo(n = 40); rng = Xoshiro(1), ntasks = 4)

    @test par.samples == serial.samples
end
