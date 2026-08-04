@testsnippet Case5 begin
    using LinearAlgebra
    net = case5()
    backend = ReferenceBackend(net)
end

@testitem "Ybus construction" tags = [:unit, :fast] setup = [Case5] begin
    Y = net.Y
    @test size(Y) == (5, 5)
    @test Y == transpose(Y)
    # Off-diagonal = minus the series admittance of the connecting branch.
    y12 = inv(complex(0.02, 0.06))
    @test Y[1, 2] ≈ -y12
    @test Y[3, 4] ≈ -inv(complex(0.01, 0.03))
    @test Y[1, 4] == 0  # no branch 1-4
    # Diagonal = incident series admittances + half line charging.
    y13 = inv(complex(0.08, 0.24))
    @test Y[1, 1] ≈ y12 + y13 + im * (0.06 / 2 + 0.05 / 2)
end

@testitem "NR self-consistency on case5" tags = [:unit, :fast] setup = [Case5] begin
    state = init_state(backend, ComponentRef[])
    info = solve!(state, backend)
    @test info.converged
    @test info.iterations <= 10
    @test info.residual < backend.tol

    # Recomputed injections S = V ∘ conj(YV) must match the specification.
    V = state.vm .* cis.(state.va)
    S = V .* conj.(net.Y * V)
    for i = 1:5
        if net.bustype[i] == 1      # PQ: P and Q enforced
            @test real(S[i]) ≈ net.pg[i] - net.pd[i] atol = 1e-8
            @test imag(S[i]) ≈ -net.qd[i] atol = 1e-8
        elseif net.bustype[i] == 2  # PV: P and vm enforced
            @test real(S[i]) ≈ net.pg[i] - net.pd[i] atol = 1e-8
            @test state.vm[i] == net.vm_setpoint[i]
        end
    end
    @test state.vm[1] == net.vm_setpoint[1]
    @test state.va[1] == 0.0

    # Slack picks up the load not covered by bus 2 plus the losses, and the sum of
    # all net injections is exactly the (positive) system losses.
    @test real(S[1]) > sum(net.pd) - sum(net.pg)
    @test sum(real(S)) > 0
end

@testitem "degenerate flat case" tags = [:unit, :fast] begin
    # Zero injections AND zero line charging (charging injects reactive power even
    # at no load, so it must be stripped for the flat start to be the solution).
    net = case5(load_scale = 0.0)
    branches = [Branch(br.f, br.t, br.y, 0.0) for br in net.branches]
    flat = NetworkData(net.bustype, net.pd, net.qd, zero(net.pg), ones(5), branches)
    backend = ReferenceBackend(flat)
    state = init_state(backend, ComponentRef[])
    info = solve!(state, backend)
    @test info.converged
    @test info.iterations == 0  # the flat start is already the solution
    @test all(state.vm .≈ 1.0)
    @test all(abs.(state.va) .< 1e-12)
end

@testitem "contract errors on bad refs" tags = [:unit, :fast] setup = [Case5] begin
    @test_throws ArgumentError init_state(backend, [ComponentRef(:load, "9", :pd)])
    @test_throws ArgumentError init_state(backend, [ComponentRef(:load, "x", :pd)])
    @test_throws ArgumentError init_state(backend, [ComponentRef(:load, "3", :bogus)])
    @test_throws ArgumentError init_state(backend, [ComponentRef(:shunt, "3", :pd)])
    # Injections at the slack bus, and reactive load at a PV bus, would be
    # silently ignored by the solve — the backend must refuse them loudly.
    @test_throws ArgumentError init_state(backend, [ComponentRef(:load, "1", :pd)])
    @test_throws ArgumentError init_state(backend, [ComponentRef(:load, "2", :qd)])
end

@testitem "set_injections! moves the solution" tags = [:unit, :fast] setup = [Case5] begin
    state = init_state(backend, [ComponentRef(:load, "5", :pd)])
    solve!(state, backend)
    v_base = extract(state, backend, VoltageMagnitude(5))
    set_injections!(state, backend, [1.2])  # double the load at bus 5
    info = solve!(state, backend)
    @test info.converged
    @test extract(state, backend, VoltageMagnitude(5)) < v_base
end

@testitem "extract QoIs" tags = [:unit, :fast] setup = [Case5] begin
    state = init_state(backend, ComponentRef[])
    solve!(state, backend)
    @test extract(state, backend, VoltageMagnitude(2)) == net.vm_setpoint[2]
    @test extract(state, backend, VoltageAngle(3)) < 0
    # Branch flows out of both ends sum to the (positive) branch loss.
    p13 = extract(state, backend, BranchActivePower(1, 3))
    p31 = extract(state, backend, BranchActivePower(3, 1))
    @test p13 > 0  # power flows from the slack toward the load
    @test p13 + p31 > 0
    @test p13 + p31 < 0.05
    @test_throws ArgumentError extract(state, backend, BranchActivePower(1, 4))
    # ViolationEvent comes free from the generic fallback.
    vm5 = extract(state, backend, VoltageMagnitude(5))
    @test extract(state, backend, ViolationEvent(VoltageMagnitude(5), 0.95, 1.05)) == 0.0
    @test extract(state, backend, ViolationEvent(VoltageMagnitude(5), vm5 + 0.01, 2.0)) ==
          1.0
end

@testitem "warm start" tags = [:unit, :fast] setup = [Case5] begin
    @test supports_warmstart(backend)
    ref = init_state(backend, [ComponentRef(:load, "3", :pd)])
    cold = solve!(ref, backend)
    state = init_state(backend, [ComponentRef(:load, "3", :pd)])
    set_injections!(state, backend, [0.47])  # nearby operating point
    warm = solve!(state, backend; warmstart = ref)
    @test warm.converged
    @test warm.iterations <= cold.iterations
    @test_throws ArgumentError solve!(state, backend; warmstart = 1.0)
end

@testitem "divergence is data, not an exception" tags = [:unit, :fast] begin
    backend = ReferenceBackend(case5(load_scale = 20.0))
    state = init_state(backend, ComponentRef[])
    info = solve!(state, backend)
    @test !info.converged
    @test info.iterations >= 0
    @test isfinite(info.residual) || isnan(info.residual)  # never throws

    # Restartability: a cold solve on a sane case still works afterwards.
    sane = ReferenceBackend(case5())
    state2 = init_state(sane, ComponentRef[])
    @test solve!(state2, sane).converged
end
