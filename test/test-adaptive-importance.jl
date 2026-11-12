@testsnippet LinearOracle begin
    using Distributions: Normal, cdf
    using ImportanceSamplers: ImportanceSamplers
    using Random: Xoshiro
    using Statistics: mean

    # A backend whose quantity is aᵀx with |a| = 1. With independent standard normal
    # injections, P(aᵀx > β) = Φ(-β) exactly. `fails(x)` marks diverged solves.
    struct LinearBackend{F} <: AbstractPFBackend
        a::Vector{Float64}
        fails::F
    end

    mutable struct LinearState
        x::Vector{Float64}
    end

    ProbabilisticPowerFlow.init_state(::LinearBackend, refs) = LinearState(zeros(length(refs)))

    function ProbabilisticPowerFlow.set_injections!(s::LinearState, ::LinearBackend, x)
        s.x .= x
        return s
    end

    function ProbabilisticPowerFlow.solve!(s::LinearState, b::LinearBackend; warmstart = nothing)
        if b.fails(s.x)
            return SolveInfo(false, -1, Inf)
        end
        return SolveInfo(true, 1, 0.0)
    end

    ProbabilisticPowerFlow.extract(s::LinearState, b::LinearBackend, ::VoltageMagnitude) =
        sum(b.a .* s.x)

    const ORACLE_D = 3
    const ORACLE_A = fill(1 / sqrt(ORACLE_D), ORACLE_D)
    const ORACLE_Y = VoltageMagnitude(1)

    function oracle(event; fails = x -> false)
        vars = [GermVariable("x$(k)", Normal()) for k in 1:ORACLE_D]
        assigns = [Assignment("x$(k)", ComponentRef(ComponentField.Pd, k)) for k in 1:ORACLE_D]
        model = UncertaintyModel(vars, assigns)
        return PPFProblem(LinearBackend(ORACLE_A, fails), model, [event])
    end

    Φ(x) = cdf(Normal(), x)
end

@testitem "AdaptiveDesignPointSampling validates its arguments" tags = [:unit, :fast] begin
    using ImportanceSamplers: ImportanceSamplers

    over = ViolationEvent(VoltageMagnitude(1), -Inf, 1.1)
    m = AdaptiveDesignPointSampling(over)
    @test (m.n, m.rounds, m.round_size, m.defensive) == (1000, 5, 200, 0.1)
    @test m.shift === nothing
    @test m.proposal === nothing
    @test AdaptiveDesignPointSampling(over; shift = [[1, 2]]).shift == [[1.0, 2.0]]
    @test AdaptiveDesignPointSampling(over; defensive = 0).defensive == 0.0

    @test_throws ArgumentError AdaptiveDesignPointSampling(over; n = 0)
    @test_throws ArgumentError AdaptiveDesignPointSampling(over; rounds = 0)
    @test_throws ArgumentError AdaptiveDesignPointSampling(over; round_size = 0)
    @test_throws ArgumentError AdaptiveDesignPointSampling(over; defensive = 1)
    @test_throws ArgumentError AdaptiveDesignPointSampling(over; defensive = -0.1)
    @test_throws ArgumentError AdaptiveDesignPointSampling(over; maxiter = 0)
    @test_throws ArgumentError AdaptiveDesignPointSampling(over; fd_step = 0)
    @test_throws ArgumentError AdaptiveDesignPointSampling(over; tol = -1.0)
    @test_throws ArgumentError AdaptiveDesignPointSampling(ViolationEvent(VoltageMagnitude(1), -Inf, Inf))
    @test_throws MethodError AdaptiveDesignPointSampling(over; shift = [1.0, 2.0])

    @test sprint(show, m) ==
        "AdaptiveDesignPointSampling(ViolationEvent(VoltageMagnitude(1), -Inf, 1.1), n = 1000, rounds = 5, round_size = 200)"
    @test endswith(sprint(show, AdaptiveDesignPointSampling(over; proposal = :bank)), ", fitted)")
end

@testitem "AdaptiveDesignPointSampling checks the stored stages against the event" tags =
    [:unit, :fast] setup = [LinearOracle] begin
    event = ViolationEvent(ORACLE_Y, -Inf, 4.0)
    prob = oracle(event)
    zstar = 4 .* ORACLE_A

    @test_throws ArgumentError solve(prob, AdaptiveDesignPointSampling(event; proposal = [:q]))
    @test_throws ArgumentError solve(prob, AdaptiveDesignPointSampling(event; shift = [zstar, zstar]))
    @test_throws DimensionMismatch solve(prob, AdaptiveDesignPointSampling(event; shift = [[4.0]]))
    @test_throws ArgumentError solve(
        prob, AdaptiveDesignPointSampling(ViolationEvent(VoltageAngle(1), -Inf, 1.0)),
    )
end

@testitem "AdaptiveDesignPointSampling matches Φ(-β) on a linear limit state" tags =
    [:unit, :fast] setup = [LinearOracle] begin
    for β in (3.0, 4.0, 5.0)
        event = ViolationEvent(ORACLE_Y, -Inf, β)
        r = solve(oracle(event), AdaptiveDesignPointSampling(event); rng = Xoshiro(1))
        p, se = violation_probability(r, event), standard_error(r, event)

        @test abs(p - Φ(-β)) < 4 * se
        # the precision does not depend on how rare the event is
        @test se / Φ(-β) < 0.05
        @test effective_sample_size(r, event) > 300
        @test r.method.shift[1] ≈ β .* ORACLE_A atol = 1.0e-3
        @test r.n_samples == 1000
        # the design point search, 5 rounds of 200, and the final samples
        @test 2000 < r.n_solves < 2100
    end
end

@testitem "AdaptiveDesignPointSampling covers both sides of a two-sided event" tags =
    [:unit, :fast] setup = [LinearOracle] begin
    event = ViolationEvent(ORACLE_Y, -4.0, 4.0)
    r = solve(oracle(event), AdaptiveDesignPointSampling(event); rng = Xoshiro(1))
    p, se = violation_probability(r, event), standard_error(r, event)

    @test length(r.method.shift) == 2
    @test length(r.method.proposal) == 2
    @test r.method.shift[1] ≈ -4 .* ORACLE_A atol = 1.0e-3
    @test r.method.shift[2] ≈ 4 .* ORACLE_A atol = 1.0e-3
    @test abs(p - 2 * Φ(-4)) < 4 * se
    @test se / (2 * Φ(-4)) < 0.05
end

@testitem "The standard error of AdaptiveDesignPointSampling has nominal coverage" tags =
    [:unit] setup = [LinearOracle] begin
    event = ViolationEvent(ORACLE_Y, -Inf, 3.0)
    prob = oracle(event)
    covered = map(1:200) do seed
        r = solve(prob, AdaptiveDesignPointSampling(event; n = 500); rng = Xoshiro(seed))
        abs(violation_probability(r, event) - Φ(-3)) <= 1.96 * standard_error(r, event)
    end
    @test 0.9 <= mean(covered) <= 0.99
end

@testitem "AdaptiveDesignPointSampling falls back when AMIS finds no violation" tags =
    [:unit, :fast] setup = [LinearOracle] begin
    event = ViolationEvent(ORACLE_Y, -Inf, 4.0)
    # a design point on the wrong side, 8 standard deviations from the violation region
    method = AdaptiveDesignPointSampling(event; n = 200, shift = [-4 .* ORACLE_A])
    r = @test_logs (:warn, r"AMIS found no violation") solve(oracle(event), method; rng = Xoshiro(1))

    @test violation_probability(r, event) == 0
    @test effective_sample_size(r, event) == 0
    # AMIS stops after its first round, then the final 200 samples
    @test r.n_solves == 200 + 200
end

@testitem "AdaptiveDesignPointSampling falls back when the proposal collapses" tags =
    [:unit, :fast] setup = [LinearOracle] begin
    event = ViolationEvent(ORACLE_Y, -Inf, 4.0)
    method = AdaptiveDesignPointSampling(event; round_size = 4)
    r = @test_logs (:warn, r"round_size = 4") (:warn, r"collapsed") solve(
        oracle(event), method; rng = Xoshiro(1),
    )

    # the fallback is DesignPointSampling, still unbiased
    p, se = violation_probability(r, event), standard_error(r, event)
    @test abs(p - Φ(-4)) < 4 * se
end

@testitem "AdaptiveDesignPointSampling records power flow failures" tags =
    [:unit, :fast] setup = [LinearOracle] begin
    event = ViolationEvent(ORACLE_Y, -Inf, 4.0)
    # failures across the violation region but off the search path, which has x1 = x2.
    # x1 - x2 is independent of aᵀx, so P(violation and converged) = Φ(-4) Φ(1.5 / √2).
    prob = oracle(event; fails = x -> x[1] - x[2] > 1.5)
    r = @test_logs solve(prob, AdaptiveDesignPointSampling(event); rng = Xoshiro(1))
    p, se = violation_probability(r, event), standard_error(r, event)

    @test !isempty(r.failures)
    @test n_converged(r) + length(r.failures) == r.n_samples
    @test all(f -> f.injections[1] - f.injections[2] > 1.5, r.failures)
    @test abs(p - Φ(-4) * Φ(1.5 / sqrt(2))) < 4 * se
end

@testitem "A fitted AdaptiveDesignPointSampling reruns only the final stage" tags =
    [:unit, :fast] setup = [LinearOracle] begin
    event = ViolationEvent(ORACLE_Y, -Inf, 4.0)
    prob = oracle(event)
    fitted = solve(prob, AdaptiveDesignPointSampling(event; n = 300); rng = Xoshiro(1)).method

    a = @test_logs solve(prob, fitted; rng = Xoshiro(2))
    b = solve(prob, fitted; rng = Xoshiro(2))
    @test a.n_solves == 300
    @test a.method.shift == fitted.shift
    @test a.method.proposal === fitted.proposal
    @test a.samples == b.samples
    @test a.weights == b.weights
    @test abs(violation_probability(a, event) - Φ(-4)) < 4 * standard_error(a, event)
end

@testitem "AdaptiveDesignPointSampling agrees with DesignPointSampling on case5" tags =
    [:integration, :powermodels] setup = [PMCase5, ISCase5] begin
    using ImportanceSamplers: ImportanceSamplers

    data = pm_case5()
    v0 = median_vm(is_problem(data, [IS_VM]))
    event = ViolationEvent(IS_VM, -Inf, v0 + 0.006)
    prob = is_problem(data, [IS_VM, event])

    plain = solve(prob, DesignPointSampling(event; n = 1000); rng = Xoshiro(1))
    adapted = @test_logs solve(prob, AdaptiveDesignPointSampling(event; n = 1000); rng = Xoshiro(1))
    p, se = violation_probability(plain, event), standard_error(plain, event)
    q, se_q = violation_probability(adapted, event), standard_error(adapted, event)

    @test abs(p - q) < 4 * sqrt(se^2 + se_q^2)
    # the adapted proposal fits the violation region better than N(z*, I)
    @test se_q < se
    @test adapted.method.shift[1] ≈ plain.method.shift atol = 1.0e-2
end
