@testsnippet ControlCase5 begin
    const STEPS = [0.0, 0.1, 0.2, 0.3]

    # case5 with an empty shunt at bus 5, for a SwitchedShunt to take over
    function shunt_case5()
        data = pm_case5()
        data["shunt"]["1"] = Dict{String, Any}(
            "index" => 1,
            "shunt_bus" => 5,
            "gs" => 0.0,
            "bs" => 0.0,
            "status" => 1,
        )
        return data
    end

    # case5 with the reactive limit of generator 2 tight enough to bind
    function limited_case5(qmax = 0.05)
        data = pm_case5()
        data["gen"]["$(gen_at(data, 2))"]["qmax"] = qmax
        return data
    end

    const MODELS = (Complementarity(), Sigmoid())

    settings(model; kwargs...) = ControlSettings(; model = model, kwargs...)

    # Whether the generator's output and its bus voltage satisfy the switching rule
    function complementarity_holds(state, b, gen, bus; atol = 1.0e-6)
        data = b.data
        qmin = data["gen"]["$(gen)"]["qmin"]
        qmax = data["gen"]["$(gen)"]["qmax"]
        vset = data["bus"]["$(bus)"]["vm"]
        q = extract(state, b, GeneratorReactivePower(gen))
        v = extract(state, b, VoltageMagnitude(bus))
        if !(qmin - atol <= q <= qmax + atol)
            return false
        elseif q >= qmax - atol
            return v <= vset + atol
        elseif q <= qmin + atol
            return v >= vset - atol
        end
        return abs(v - vset) <= atol
    end

    function finite_difference_jacobian(sys, x)
        n = length(x)
        J = zeros(n, n)
        Fp = similar(x)
        Fm = similar(x)
        h = 1.0e-7
        for k in 1:n
            xp = copy(x)
            xm = copy(x)
            xp[k] += h
            xm[k] -= h
            sys.f!(Fp, xp, sys.p0)
            sys.f!(Fm, xm, sys.p0)
            J[:, k] = (Fp - Fm) / 2h
        end
        return J
    end
end

@testitem "Control settings and switched shunts validate their input" tags =
    [:unit, :fast] begin
    s = ControlSettings()
    @test s.model isa Complementarity
    @test s.model.scale === nothing
    @test s.snap
    @test Complementarity(scale = 0.05).scale == 0.05
    @test_throws ArgumentError Complementarity(scale = 0.0)
    @test_throws ArgumentError Complementarity(scale = Inf)

    m = Sigmoid()
    @test m.steepness == 5000.0
    @test m.homotopy == [100.0, 1000.0]
    @test Sigmoid(homotopy = Float64[]).homotopy == Float64[]
    @test_throws ArgumentError Sigmoid(steepness = 0.0)
    @test_throws ArgumentError Sigmoid(steepness = Inf)
    @test_throws ArgumentError Sigmoid(homotopy = [1000.0, 100.0])
    @test_throws ArgumentError Sigmoid(homotopy = [100.0, 100.0])
    @test_throws ArgumentError Sigmoid(homotopy = [-1.0, 100.0])
    @test_throws ArgumentError Sigmoid(steepness = 500.0, homotopy = [100.0, 1000.0])

    sh = SwitchedShunt(3; steps = [0.2, 0.0, 0.1, 0.1], vset = 1.02)
    @test sh.steps == [0.0, 0.1, 0.2]
    @test sh.vset == 1.02
    @test_throws ArgumentError SwitchedShunt(3; steps = Float64[], vset = 1.0)
    @test_throws ArgumentError SwitchedShunt(3; steps = [0.0, Inf], vset = 1.0)
    @test_throws ArgumentError SwitchedShunt(3; steps = [0.0], vset = 0.0)
end

@testitem "The backend rejects controls it cannot honour" tags =
    [:integration, :powermodels] setup = [PMCase5, ControlCase5] begin
    data = shunt_case5()
    slack_gen = gen_at(data, 1)
    pv_gen = gen_at(data, 2)

    b = PowerModelsBackend(data; controls = [ReactiveLimits(pv_gen)])
    @test b.controls == [ReactiveLimits(pv_gen)]
    @test isempty(PowerModelsBackend(data).controls)

    @test_throws ArgumentError PowerModelsBackend(data; controls = [ReactiveLimits(99)])
    @test_throws ArgumentError PowerModelsBackend(data; controls = [ReactiveLimits(slack_gen)])
    @test_throws ArgumentError PowerModelsBackend(
        data;
        controls = [ReactiveLimits(pv_gen), ReactiveLimits(pv_gen)],
    )

    nolimit = deepcopy(data)
    nolimit["gen"]["$(pv_gen)"]["qmax"] = Inf
    @test_throws ArgumentError PowerModelsBackend(nolimit; controls = [ReactiveLimits(pv_gen)])

    fixed = deepcopy(data)
    fixed["gen"]["$(pv_gen)"]["qmin"] = 0.1
    fixed["gen"]["$(pv_gen)"]["qmax"] = 0.1
    @test PowerModelsBackend(fixed; controls = [ReactiveLimits(pv_gen)]) isa PowerModelsBackend
    fixed["gen"]["$(pv_gen)"]["qmin"] = 0.2
    @test_throws ArgumentError PowerModelsBackend(fixed; controls = [ReactiveLimits(pv_gen)])

    # a second generator at bus 2 must be limited too
    twogen = deepcopy(data)
    twogen["gen"]["9"] = deepcopy(twogen["gen"]["$(pv_gen)"])
    twogen["gen"]["9"]["index"] = 9
    @test_throws ArgumentError PowerModelsBackend(twogen; controls = [ReactiveLimits(pv_gen)])
    @test PowerModelsBackend(twogen; controls = [ReactiveLimits(pv_gen), ReactiveLimits(9)]) isa
        PowerModelsBackend

    shunt = SwitchedShunt(1; steps = STEPS, vset = 1.02)
    @test PowerModelsBackend(data; controls = [shunt]) isa PowerModelsBackend
    @test_throws ArgumentError PowerModelsBackend(
        data;
        controls = [SwitchedShunt(7; steps = STEPS, vset = 1.02)],
    )

    atpv = deepcopy(data)
    atpv["shunt"]["1"]["shunt_bus"] = 2
    @test_throws ArgumentError PowerModelsBackend(atpv; controls = [shunt])

    off = deepcopy(data)
    off["shunt"]["1"]["status"] = 0
    @test_throws ArgumentError PowerModelsBackend(off; controls = [shunt])

    # the normal map gives each bus one control variable, the sigmoid does not need that
    two = deepcopy(data)
    two["shunt"]["2"] = deepcopy(two["shunt"]["1"])
    two["shunt"]["2"]["index"] = 2
    both = [shunt, SwitchedShunt(2; steps = STEPS, vset = 1.02)]
    @test_throws ArgumentError PowerModelsBackend(two; controls = both)
    @test PowerModelsBackend(two; controls = both, control_settings = settings(Sigmoid())) isa
        PowerModelsBackend
end

@testitem "The control Jacobian matches a finite difference" tags =
    [:integration, :powermodels] setup = [PMCase5, ControlCase5] begin
    # a limit that binds and one that does not, with every model, snapped and not
    for qmax in (0.05, 1.0), model in MODELS, snap in (false, true)
        data = shunt_case5()
        data["gen"]["$(gen_at(data, 2))"]["qmax"] = qmax
        controls =
            [ReactiveLimits(gen_at(data, 2)), SwitchedShunt(1; steps = STEPS, vset = 1.02)]
        b = PowerModelsBackend(data; controls = controls, control_settings = settings(model; snap))
        state = init_state(b, ComponentRef[])
        @test solve!(state, b).converged

        # a snapped shunt's slot holds its bus voltage, not the continuous slot value
        x = copy(state.last_solution)
        if snap
            x[2 * state.pf_data.am.bus_to_idx[5] - 1] = extract(state, b, VoltageMagnitude(5))
        end

        # near the solution but away from the kinks of the normal map
        x = x .+ 1.0e-4 .* sin.(1:length(x))
        J = copy(state.sys.jac_prototype)
        state.sys.j!(J, x, state.sys.p0)
        @test Matrix(J) ≈ finite_difference_jacobian(state.sys, x) atol = 1.0e-5 rtol = 1.0e-7
    end
end

@testitem "A binding reactive limit matches a PQ bus at that limit" tags =
    [:integration, :powermodels] setup = [PMCase5, ControlCase5] begin
    # the upper limit, the lower limit, and a fixed output
    for (qmin, qmax, q) in ((-3.0, 0.05, 0.05), (0.8, 1.0, 0.8), (0.1, 0.1, 0.1))
        data = pm_case5()
        gen = gen_at(data, 2)
        data["gen"]["$(gen)"]["qmin"] = qmin
        data["gen"]["$(gen)"]["qmax"] = qmax
        vset = data["bus"]["2"]["vm"]

        # the same network with generator 2 replaced by a fixed injection at q
        ref = deepcopy(data)
        ref["bus"]["2"]["bus_type"] = 1
        ref["gen"]["$(gen)"]["gen_status"] = 0
        load = ref["load"]["$(load_at(data, 2))"]
        load["pd"] -= data["gen"]["$(gen)"]["pg"]
        load["qd"] -= q
        sol = PowerModels.compute_ac_pf(ref)["solution"]["bus"]

        for model in MODELS
            b = PowerModelsBackend(
                data;
                controls = [ReactiveLimits(gen)],
                control_settings = settings(model),
            )
            state = init_state(b, ComponentRef[])
            @test solve!(state, b).converged

            @test extract(state, b, GeneratorReactivePower(gen)) ≈ q rtol = 1.0e-6
            if q == qmax && qmin < qmax
                @test extract(state, b, VoltageMagnitude(2)) < vset - 1.0e-3
            elseif q == qmin && qmin < qmax
                @test extract(state, b, VoltageMagnitude(2)) > vset + 1.0e-3
            end
            for bus in 1:5
                @test extract(state, b, VoltageMagnitude(bus)) ≈ sol["$(bus)"]["vm"] atol =
                    1.0e-8
                @test extract(state, b, VoltageAngle(bus)) ≈ sol["$(bus)"]["va"] atol = 1.0e-8
            end
        end
    end
end

@testitem "Reactive load at a limited PV bus can be sampled" tags =
    [:integration, :powermodels] setup = [PMCase5, ControlCase5] begin
    data = limited_case5()
    gen = gen_at(data, 2)
    ref = ComponentRef(ComponentField.Qd, load_at(data, 2))
    @test_throws ArgumentError init_state(PowerModelsBackend(data), [ref])

    for model in MODELS
        b = PowerModelsBackend(data; controls = [ReactiveLimits(gen)], control_settings = settings(model))
        state = init_state(b, [ref])
        for qd in (-0.3, 0.1, 0.4)
            set_injections!(state, b, [qd])
            @test solve!(state, b).converged

            # the same network with the reactive load fixed in the data
            fixed = deepcopy(data)
            fixed["load"]["$(load_at(data, 2))"]["qd"] = qd
            fb = PowerModelsBackend(fixed; controls = [ReactiveLimits(gen)], control_settings = settings(model))
            fs = init_state(fb, ComponentRef[])
            @test solve!(fs, fb).converged
            for bus in 1:5
                @test extract(state, b, VoltageMagnitude(bus)) ≈ extract(fs, fb, VoltageMagnitude(bus)) atol =
                    1.0e-8
            end
            @test extract(state, b, GeneratorReactivePower(gen)) ≈
                extract(fs, fb, GeneratorReactivePower(gen)) atol = 1.0e-8
        end
    end
end

@testitem "A reactive limit that does not bind keeps the setpoint" tags =
    [:integration, :powermodels] setup = [PMCase5, ControlCase5] begin
    data = limited_case5(1.0)
    gen = gen_at(data, 2)
    free = PowerModelsBackend(data)
    s_free = init_state(free, ComponentRef[])
    @test solve!(s_free, free).converged

    # complementarity is exact, the sigmoid is within a few multiples of 1 / steepness
    for (model, atol) in ((Complementarity(), 1.0e-8), (Sigmoid(), 1.0e-3))
        b = PowerModelsBackend(data; controls = [ReactiveLimits(gen)], control_settings = settings(model))
        state = init_state(b, ComponentRef[])
        @test solve!(state, b).converged

        @test extract(state, b, VoltageMagnitude(2)) ≈ data["bus"]["2"]["vm"] atol = atol
        for bus in 3:5
            @test extract(state, b, VoltageMagnitude(bus)) ≈
                extract(s_free, free, VoltageMagnitude(bus)) atol = atol
        end
        @test_throws ArgumentError extract(state, b, GeneratorReactivePower(gen_at(data, 1)))
    end
end

@testitem "The complementarity scale does not change the solution" tags =
    [:integration, :powermodels] setup = [PMCase5, ControlCase5] begin
    data = shunt_case5()
    data["gen"]["$(gen_at(data, 2))"]["qmax"] = 0.05
    controls = [ReactiveLimits(gen_at(data, 2)), SwitchedShunt(1; steps = STEPS, vset = 1.02)]
    results = map((nothing, 0.1, 1.0)) do scale
        model = Complementarity(scale = scale)
        b = PowerModelsBackend(data; controls = controls, control_settings = settings(model))
        state = init_state(b, ComponentRef[])
        @test solve!(state, b).converged
        [extract(state, b, VoltageMagnitude(bus)) for bus in 1:5]
    end
    @test results[1] ≈ results[2] atol = 1.0e-10
    @test results[1] ≈ results[3] atol = 1.0e-10
end

@testitem "Generators on one limited bus share its output" tags =
    [:integration, :powermodels] setup = [PMCase5, ControlCase5] begin
    data = limited_case5(0.02)
    gen = gen_at(data, 2)
    data["gen"]["9"] = deepcopy(data["gen"]["$(gen)"])
    data["gen"]["9"]["index"] = 9
    data["gen"]["9"]["pg"] = 0.0
    data["gen"]["9"]["qmin"] = -0.5
    data["gen"]["9"]["qmax"] = 0.04

    for model in MODELS
        b = PowerModelsBackend(
            data;
            controls = [ReactiveLimits(gen), ReactiveLimits(9)],
            control_settings = settings(model),
        )
        state = init_state(b, ComponentRef[])
        @test solve!(state, b).converged

        q1 = extract(state, b, GeneratorReactivePower(gen))
        q2 = extract(state, b, GeneratorReactivePower(9))
        @test q1 ≈ 0.02 rtol = 1.0e-6
        @test q2 ≈ 0.04 rtol = 1.0e-6
    end
end

@testitem "A switched shunt snaps to a step" tags = [:integration, :powermodels] setup =
    [PMCase5, ControlCase5] begin
    data = shunt_case5()
    shunt = SwitchedShunt(1; steps = STEPS, vset = 1.02)
    ref = ComponentRef(ComponentField.Pd, load_at(data, 5))

    for model in MODELS
        snapped = PowerModelsBackend(data; controls = [shunt], control_settings = settings(model))
        continuous = PowerModelsBackend(
            data;
            controls = [shunt],
            control_settings = settings(model; snap = false),
        )

        s_snapped = init_state(snapped, [ref])
        s_continuous = init_state(continuous, [ref])
        seen = Float64[]
        for pd in (0.3, 0.6, 1.0, 1.4)
            set_injections!(s_snapped, snapped, [pd])
            set_injections!(s_continuous, continuous, [pd])
            @test solve!(s_snapped, snapped).converged
            @test solve!(s_continuous, continuous).converged

            b_continuous = extract(s_continuous, continuous, ShuntSusceptance(1))
            @test 0.0 <= b_continuous <= 0.3
            if model isa Complementarity && 0.0 < b_continuous < 0.3
                @test extract(s_continuous, continuous, VoltageMagnitude(5)) ≈ 1.02 atol =
                    1.0e-8
            end

            b_step = extract(s_snapped, snapped, ShuntSusceptance(1))
            @test b_step in STEPS
            push!(seen, b_step)

            # the snapped solve is plain PowerModels with the shunt fixed at that step
            fixed = deepcopy(data)
            fixed["shunt"]["1"]["bs"] = b_step
            fixed["load"]["$(load_at(data, 5))"]["pd"] = pd
            sol = PowerModels.compute_ac_pf(fixed)["solution"]["bus"]
            for bus in 1:5
                @test extract(s_snapped, snapped, VoltageMagnitude(bus)) ≈
                    sol["$(bus)"]["vm"] atol = 1.0e-7
            end
        end

        # more load pulls the voltage down and switches in more susceptance
        @test issorted(seen)
        @test first(seen) < last(seen)
    end
end

@testitem "An uncontrolled shunt reports its fixed susceptance" tags =
    [:integration, :powermodels] setup = [PMCase5, ControlCase5] begin
    data = shunt_case5()
    data["shunt"]["1"]["bs"] = 0.15
    b = PowerModelsBackend(data)
    state = init_state(b, ComponentRef[])
    @test solve!(state, b).converged
    @test extract(state, b, ShuntSusceptance(1)) == 0.15
    @test_throws ArgumentError extract(state, b, ShuntSusceptance(2))
end

@testitem "A failed steep solve falls back to the homotopy" tags =
    [:integration, :powermodels] setup = [PMCase5, ControlCase5] begin
    # fails its first solve, then solves normally, and counts its calls
    mutable struct FailFirst
        calls::Int
    end
    function PowerModels._solve_nl(sys::PowerModels.PowerFlowSystem, alg::FailFirst)
        alg.calls += 1
        if alg.calls == 1
            return PowerModels.PowerFlowSolution(copy(sys.x0), false, 3, Inf)
        end
        return PowerModels._solve_nl(sys, PowerModels.NativeNewton())
    end

    data = limited_case5()
    controls = [ReactiveLimits(gen_at(data, 2))]
    sigmoid = settings(Sigmoid())
    direct = PowerModelsBackend(data; controls = controls, control_settings = sigmoid)
    s_direct = init_state(direct, ComponentRef[])
    @test solve!(s_direct, direct).converged

    alg = FailFirst(0)
    b = PowerModelsBackend(data; alg = alg, controls = controls, control_settings = sigmoid)
    state = init_state(b, ComponentRef[])
    info = solve!(state, b)
    @test info.converged
    @test alg.calls == 1 + 3
    @test info.iterations > 3
    for bus in 1:5
        @test extract(state, b, VoltageMagnitude(bus)) ≈
            extract(s_direct, direct, VoltageMagnitude(bus)) atol = 1.0e-8
    end

    alg.calls = 0
    nohomotopy = PowerModelsBackend(
        data;
        alg = alg,
        controls = controls,
        control_settings = settings(Sigmoid(homotopy = Float64[])),
    )
    @test !solve!(init_state(nohomotopy, ComponentRef[]), nohomotopy).converged
    @test alg.calls == 1

    # complementarity has no steepness to walk
    alg.calls = 0
    exact = PowerModelsBackend(data; alg = alg, controls = controls)
    @test !solve!(init_state(exact, ComponentRef[]), exact).converged
    @test alg.calls == 1
end

@testitem "Divergence with controls is returned, not thrown" tags =
    [:integration, :powermodels] setup = [PMCase5, ControlCase5] begin
    data = shunt_case5()
    controls = [ReactiveLimits(gen_at(data, 2)), SwitchedShunt(1; steps = STEPS, vset = 1.02)]
    refs = [ComponentRef(ComponentField.Pd, load_at(data, bus)) for bus in 3:5]
    for model in MODELS
        b = PowerModelsBackend(data; controls = controls, control_settings = settings(model))
        state = init_state(b, refs)

        set_injections!(state, b, [200.0, 200.0, 200.0])
        @test !solve!(state, b).converged

        set_injections!(state, b, [0.45, 0.4, 0.6])
        @test solve!(state, b).converged
        @test extract(state, b, ShuntSusceptance(1)) in STEPS
    end
end

@testitem "Complementarity holds across a load sweep" tags = [:integration, :powermodels] setup =
    [PMCase5, ControlCase5] begin
    using NonlinearSolve: NewtonRaphson

    data = shunt_case5()
    gen = gen_at(data, 2)
    data["gen"]["$(gen)"]["qmax"] = 0.3
    controls = [ReactiveLimits(gen), SwitchedShunt(1; steps = STEPS, vset = 1.0)]
    ref = ComponentRef(ComponentField.Qd, load_at(data, 5))

    # the reactive load at bus 5 drives generator 2 from inside its limits to qmax
    for alg in (PowerModels.NativeNewton(), NewtonRaphson())
        b = PowerModelsBackend(data; alg = alg, controls = controls)
        state = init_state(b, [ref])
        chain = init_state(b, [ref])
        at_limit = 0
        for qd in -0.5:0.1:1.9
            set_injections!(state, b, [qd])
            set_injections!(chain, b, [qd])
            @test solve!(state, b).converged
            @test solve!(chain, b; warmstart = chain).converged
            @test complementarity_holds(state, b, gen, 2)
            @test extract(chain, b, VoltageMagnitude(2)) ≈
                extract(state, b, VoltageMagnitude(2)) atol = 1.0e-8
            at_limit += extract(state, b, GeneratorReactivePower(gen)) ≈ 0.3
        end
        @test 0 < at_limit < length(-0.5:0.1:1.9)
    end
end

@testitem "Monte Carlo estimates switching probabilities" tags =
    [:integration, :powermodels] setup = [PMCase5, ControlCase5] begin
    using Distributions: Normal
    using Random: Xoshiro

    data = shunt_case5()
    b = PowerModelsBackend(
        data;
        controls = [ReactiveLimits(gen_at(data, 2)), SwitchedShunt(1; steps = STEPS, vset = 1.02)],
    )
    ref = ComponentRef(ComponentField.Pd, load_at(data, 5))
    model = UncertaintyModel([GermVariable("pd5", Normal(0.8, 0.3))], [Assignment("pd5", ref)])
    qois = [ShuntSusceptance(1), GeneratorReactivePower(gen_at(data, 2)), VoltageMagnitude(5)]
    prob = PPFProblem(b, model, qois)

    off = solve(prob, MonteCarlo(n = 200); rng = Xoshiro(3))
    chain = solve(prob, MonteCarlo(n = 200); rng = Xoshiro(3), warmstart = :chain)
    @test failure_rate(off) == 0

    steps = qoi_samples(off, ShuntSusceptance(1))
    @test all(in(STEPS), steps)
    probabilities = [count(==(s), steps) / length(steps) for s in STEPS]
    @test sum(probabilities) ≈ 1.0
    @test count(>(0), probabilities) >= 2

    @test qoi_samples(chain, ShuntSusceptance(1)) == steps
    @test qoi_samples(chain, VoltageMagnitude(5)) ≈ qoi_samples(off, VoltageMagnitude(5)) atol =
        1.0e-6
end
