@testsnippet Stub begin
    using Distributions
    using Copulas
    using Random: Xoshiro
    using Statistics: mean, std
    using ProbabilisticPowerFlow: AffineTransform

    # A backend with no power flow in it. Every "voltage" is an affine function of the
    # injections, which makes the loop's bookkeeping checkable by hand, and a solve
    # "diverges" once the total injection passes `limit`, which is how the failure
    # paths get exercised without a genuinely hard network.
    struct Stub <: AbstractPFBackend
        limit::Float64
        warm::Bool
    end
    Stub(; limit = Inf, warm = true) = Stub(limit, warm)

    mutable struct StubState
        x::Vector{Float64}
        warmed::Int
        solves::Int
    end

    ProbabilisticPowerFlow.init_state(::Stub, refs) = StubState(zeros(length(refs)), 0, 0)

    function ProbabilisticPowerFlow.set_injections!(s::StubState, ::Stub, x)
        copyto!(s.x, x)
        return s
    end

    function ProbabilisticPowerFlow.solve!(s::StubState, b::Stub; warmstart = nothing)
        s.solves += 1
        warmstart === nothing || (s.warmed += 1)
        sum(s.x) > b.limit && return SolveInfo(false, 3, Inf)
        return SolveInfo(true, 2, 1e-12)
    end

    ProbabilisticPowerFlow.extract(s::StubState, ::Stub, q::VoltageMagnitude) =
        1.0 - 0.1 * s.x[q.bus]
    ProbabilisticPowerFlow.supports_warmstart(b::Stub) = b.warm

    VARIABLES = [GermVariable("a", Normal(1.0, 0.2)), GermVariable("b", Uniform(0.0, 1.0))]
    ASSIGNMENTS = [
        Assignment("a", ComponentRef(ComponentField.Pd, 1)),
        Assignment("b", ComponentRef(ComponentField.Pd, 2), AffineTransform(2.0, 0.0)),
    ]
    MODEL(rho = 0.0) =
        UncertaintyModel(VARIABLES, ASSIGNMENTS, GaussianCopula([1.0 rho; rho 1.0]))

    QOIS = [VoltageMagnitude(1), VoltageMagnitude(2)]
    problem(; limit = Inf, warm = true, rho = 0.0) =
        PPFProblem(Stub(; limit, warm), MODEL(rho), QOIS)
end

@testitem "A run accounts for every sample in the budget" setup = [Stub] tags =
    [:unit, :fast] begin
    r = solve(problem(), MonteCarlo(n = 50); rng = Xoshiro(1))

    @test r.n_samples == 50
    @test r.n_solves == 50
    @test n_converged(r) + length(r.failures) == 50
    @test n_converged(r) == 50
    @test failure_rate(r) == 0.0
    @test size(r.samples) == (2, 50)
    @test r.sample_indices == 1:50
    @test r.u === nothing
    @test r.weights === nothing
end

@testitem "The same seed gives the same run" setup = [Stub] tags = [:unit, :fast] begin
    p = problem()
    a = solve(p, MonteCarlo(n = 30); rng = Xoshiro(7))
    b = solve(p, MonteCarlo(n = 30); rng = Xoshiro(7))
    c = solve(p, MonteCarlo(n = 30); rng = Xoshiro(8))

    @test a.samples == b.samples
    @test a.samples != c.samples
end

@testitem "Concurrency changes the schedule, not the answer" setup = [Stub] tags =
    [:unit, :fast] begin
    p = problem()
    serial = solve(p, MonteCarlo(n = 40); rng = Xoshiro(3))
    par = solve(p, MonteCarlo(n = 40); rng = Xoshiro(3), ntasks = 4)

    # not merely equal in distribution: the draws happen before the solves and every
    # task gets its own state, so cold parallel runs are identical to serial ones
    @test par.samples == serial.samples
    @test par.sample_indices == serial.sample_indices
end

@testitem ":sorted permutes the solve order but not the sample set" setup = [Stub] tags =
    [:unit, :fast] begin
    p = problem()
    off = solve(p, MonteCarlo(n = 40, warmstart = :off); rng = Xoshiro(11))
    sorted = solve(p, MonteCarlo(n = 40, warmstart = :sorted); rng = Xoshiro(11))

    @test sorted.sample_indices != off.sample_indices
    @test sort(sorted.sample_indices) == off.sample_indices
    # column j of sorted came from draw sample_indices[j], so undoing the permutation
    # has to reproduce the draw-ordered run exactly
    @test sorted.samples[:, sortperm(sorted.sample_indices)] == off.samples
end

@testitem "Divergence is recorded, never dropped" setup = [Stub] tags = [:unit, :fast] begin
    p = problem(limit = 1.6)
    r = solve(p, MonteCarlo(n = 200); rng = Xoshiro(5))

    @test !isempty(r.failures)
    @test n_converged(r) + length(r.failures) == 200
    @test failure_rate(r) ≈ length(r.failures) / 200
    @test all(f -> sum(f.injections) > 1.6, r.failures)
    @test all(f -> !f.info.converged, r.failures)
    @test all(f -> 1 <= f.index <= 200, r.failures)
    @test length(unique(f.index for f in r.failures)) == length(r.failures)

    # a failure keeps its own u point even though keep_inputs is off
    @test all(f -> length(f.u) == 2, r.failures)
    @test all(f -> all(0 .< f.u .< 1), r.failures)
end

@testitem "keep_inputs round-trips u back to x" setup = [Stub] tags = [:unit, :fast] begin
    p = problem()
    r = solve(p, MonteCarlo(n = 20, keep_inputs = true); rng = Xoshiro(2))

    @test size(r.u) == (2, 20)
    # samples are a deterministic function of u, so replaying a stored column has to
    # reproduce the recorded QoI exactly
    for j = 1:20
        x = to_physical(p.model, r.u[:, j])
        @test r.samples[1, j] == 1.0 - 0.1 * x[1]
        @test r.samples[2, j] == 1.0 - 0.1 * x[2]
    end
end

@testitem "Warm starts are used, and only where they exist" setup = [Stub] tags =
    [:unit, :fast] begin
    p = problem()
    off = solve(p, MonteCarlo(n = 10, warmstart = :off); rng = Xoshiro(4))
    chain = solve(p, MonteCarlo(n = 10, warmstart = :chain); rng = Xoshiro(4))

    # the stub solves exactly, so warm starting must not move the answer
    @test chain.samples ≈ off.samples

    cold = PPFProblem(Stub(warm = false), MODEL(), QOIS)
    @test_throws ArgumentError solve(cold, MonteCarlo(n = 5, warmstart = :chain))
    @test_throws ArgumentError solve(cold, MonteCarlo(n = 5, warmstart = :sorted))
    # :off asks nothing of the backend, so it still runs
    @test n_converged(solve(cold, MonteCarlo(n = 5))) == 5
end

@testitem "Method arguments are checked before any solve" setup = [Stub] tags =
    [:unit, :fast] begin
    p = problem()

    @test_throws ArgumentError solve(p, MonteCarlo(n = 5, warmstart = :warm))
    @test_throws ArgumentError solve(p, MonteCarlo(n = 5, failure_policy = :retry))
    @test_throws ArgumentError solve(p, MonteCarlo(n = 5); ntasks = 0)
end

@testitem "PPFProblem accepts a heterogeneous QoI list" setup = [Stub] tags = [:unit, :fast] begin
    qois = [VoltageMagnitude(1), ViolationEvent(VoltageMagnitude(2), 0.95, 1.05)]
    p = PPFProblem(Stub(), MODEL(), qois)

    @test p.qois isa Vector{AbstractQoI}
    @test length(p.qois) == 2

    r = solve(p, MonteCarlo(n = 25); rng = Xoshiro(6))
    @test all(v -> v in (0.0, 1.0), r.samples[2, :])
end
