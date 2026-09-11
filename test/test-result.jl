@testsnippet Results begin
    using Statistics: mean, std, quantile

    struct NoMethod <: AbstractPPFMethod end

    const vm = VoltageMagnitude(2)
    const band = ViolationEvent(vm, 0.95, 1.05)

    function example_result(qois = AbstractQoI[vm, VoltageAngle(2)])
        samples = [
            1.0 0.94 1.06
            0.1 0.2 0.3
        ]
        failure = FailedSample(3, [0.5], [2.0], SolveInfo(false, -1, Inf))
        return PPFResult(NoMethod(), qois, samples, [1, 2, 4], [failure], 4, 4)
    end
end

@testitem "PPFResult counts converged and failed samples" setup = [Results] tags = [:unit, :fast] begin
    r = example_result()

    @test n_converged(r) == 3
    @test failure_rate(r) == 0.25
    @test n_converged(r) + length(r.failures) == r.n_samples
end

@testitem "Statistics read the row of a QoI" setup = [Results] tags = [:unit, :fast] begin
    r = example_result()

    @test qoi_samples(r, vm) == [1.0, 0.94, 1.06]
    @test mean(r, VoltageAngle(2)) ≈ 0.2
    @test std(r, vm) ≈ std([1.0, 0.94, 1.06])
    @test quantile(r, vm, 0.5) ≈ 1.0
    @test_throws ArgumentError qoi_samples(r, VoltageMagnitude(3))
end

@testitem "A ViolationEvent is derived from its quantity" setup = [Results] tags = [:unit, :fast] begin
    r = example_result()

    @test qoi_samples(r, band) == [0.0, 1.0, 1.0]
    @test violation_probability(r, band) ≈ 2 / 3

    estimated = example_result(AbstractQoI[vm, band])
    @test qoi_samples(estimated, band) == [0.1, 0.2, 0.3]

    @test_throws ArgumentError qoi_samples(r, ViolationEvent(VoltageMagnitude(3), 0.9, 1.1))
end

@testitem "PPFProblem accepts a vector of a single QoI type" tags = [:unit, :fast] begin
    using Distributions: Normal

    struct NoBackend <: AbstractPFBackend end

    model = UncertaintyModel(
        [GermVariable("load", Normal(1.0, 0.1))],
        [Assignment("load", ComponentRef(ComponentField.Pd, 1))],
    )
    prob = PPFProblem(NoBackend(), model, [VoltageMagnitude(1)])

    @test prob.qois isa Vector{AbstractQoI}
    @test isconcretetype(typeof(prob))
end
