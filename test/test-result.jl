@testsnippet Results begin
    using Statistics: mean, std, quantile
    using ProbabilisticPowerFlow: qoi_index

    QOIS = AbstractQoI[VoltageMagnitude(1), VoltageMagnitude(2)]

    # A result built by hand. Everything result.jl does is arithmetic on this matrix,
    # so no solver is needed to check it.
    function result(;
        samples = [1.00 0.94 1.02; 0.99 1.06 1.00],
        indices = [1, 2, 4],
        failures = [FailedSample(3, [0.5, 0.5], [1.0, 2.0], SolveInfo(false, 3, Inf))],
        u = nothing,
        weights = nothing,
        n = 4,
    )
        return PPFResult(
            MonteCarlo(n = n),
            QOIS,
            samples,
            indices,
            failures,
            u,
            weights,
            n,
            n,
        )
    end
end

@testitem "The budget invariant holds" setup = [Results] tags = [:unit, :fast] begin
    r = result()

    @test n_converged(r) == 3
    @test length(r.failures) == 1
    @test n_converged(r) + length(r.failures) == r.n_samples
    @test failure_rate(r) == 0.25
    @test failure_rate(result(failures = FailedSample[], indices = [1, 2, 3], n = 3)) == 0.0
end

@testitem "A QoI that was not estimated is named, not indexed" setup = [Results] tags =
    [:unit, :fast] begin
    r = result()

    @test qoi_index(r, VoltageMagnitude(1)) == 1
    @test qoi_index(r, VoltageMagnitude(2)) == 2
    @test_throws ArgumentError qoi_index(r, VoltageMagnitude(3))
    @test_throws ArgumentError qoi_samples(r, VoltageAngle(1))
end

@testitem "Statistics read the converged samples only" setup = [Results] tags =
    [:unit, :fast] begin
    r = result()

    @test qoi_samples(r, VoltageMagnitude(1)) == [1.00, 0.94, 1.02]
    @test mean(r, VoltageMagnitude(1)) ≈ mean([1.00, 0.94, 1.02])
    @test std(r, VoltageMagnitude(2)) ≈ std([0.99, 1.06, 1.00])
    @test quantile(r, VoltageMagnitude(1), 0.5) ≈ 1.00
end

@testitem "A violation band can be applied after the run" setup = [Results] tags =
    [:unit, :fast] begin
    r = result()
    v = ViolationEvent(VoltageMagnitude(1), 0.95, 1.05)

    # the event was not estimated, so it is derived from its own quantity: no solves
    @test qoi_samples(r, v) == [0.0, 1.0, 0.0]
    @test violation_probability(r, v) ≈ 1 / 3

    # a second band on the same quantity costs nothing either
    @test violation_probability(r, ViolationEvent(VoltageMagnitude(1), 0.0, 2.0)) == 0.0
    @test violation_probability(r, ViolationEvent(VoltageMagnitude(2), 0.0, 1.0)) ≈ 1 / 3

    # but only if the quantity itself was recorded
    @test_throws ArgumentError qoi_samples(r, ViolationEvent(VoltageAngle(1), 0.0, 1.0))
end

@testitem "An estimated event is read, not recomputed" setup = [Results] tags =
    [:unit, :fast] begin
    v = ViolationEvent(VoltageMagnitude(1), 0.95, 1.05)
    r = PPFResult(
        MonteCarlo(n = 3),
        AbstractQoI[VoltageMagnitude(1), v],
        [1.00 0.94 1.02; 0.0 1.0 0.0],
        [1, 2, 3],
        FailedSample[],
        nothing,
        nothing,
        3,
        3,
    )

    @test qoi_samples(r, v) === view(r.samples, 2, :)
    @test violation_probability(r, v) ≈ 1 / 3
end

@testitem "Weights are self-normalized" setup = [Results] tags = [:unit, :fast] begin
    x = [1.00, 0.94, 1.02]
    w = [0.5, 2.0, 1.5]
    r = result(weights = w)

    @test mean(r, VoltageMagnitude(1)) ≈ sum(w .* x) / sum(w)
    # weights that do not average to one still give an unbiased estimate
    @test mean(result(weights = 10 .* w), VoltageMagnitude(1)) ≈
          mean(r, VoltageMagnitude(1))
    # uniform weights reduce to the plain mean
    @test mean(result(weights = ones(3)), VoltageMagnitude(1)) ≈ mean(x)

    mu = sum(w .* x) / sum(w)
    @test std(r, VoltageMagnitude(1)) ≈ sqrt(sum(w .* (x .- mu) .^ 2) / sum(w))
end
