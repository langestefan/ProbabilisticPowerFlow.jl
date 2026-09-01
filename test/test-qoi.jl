@testsnippet Backends begin
    struct VecBackend <: AbstractPFBackend end
    struct DictBackend <: AbstractPFBackend end

    ProbabilisticPowerFlow.extract(s, ::VecBackend, q::VoltageMagnitude) = s[q.bus]
    ProbabilisticPowerFlow.extract(s, ::DictBackend, q::VoltageMagnitude) = s[q.bus]
end

@testitem "QoI leaves compare equal by value" tags=[:unit, :fast] begin
    @test VoltageMagnitude(2) == VoltageMagnitude(2)
    @test VoltageMagnitude(2) != VoltageMagnitude(3)
    @test VoltageAngle(2) == VoltageAngle(2)
    @test BranchActivePower(1, 2) == BranchActivePower(1, 2)
    @test BranchActivePower(1, 2) != BranchActivePower(2, 1)
    @test BranchReactivePower(1, 2) == BranchReactivePower(1, 2)

    @test isbitstype(VoltageMagnitude)
    @test isbitstype(ViolationEvent{VoltageMagnitude})
end

@testitem "extract on a ViolationEvent delegates to its inner QoI" setup=[Backends] tags=[
    :unit,
    :fast,
] begin
    b = VecBackend()
    v = ViolationEvent(VoltageMagnitude(2), 0.95, 1.05)

    @test extract([1.0, 1.00], b, v) == 0.0
    @test extract([1.0, 1.10], b, v) == 1.0
    @test extract([1.0, 0.90], b, v) == 1.0
end

@testitem "The violation band is closed at both ends" setup=[Backends] tags=[:unit, :fast] begin
    b = VecBackend()
    v = ViolationEvent(VoltageMagnitude(1), 0.95, 1.05)

    @test extract([0.95], b, v) == 0.0
    @test extract([1.05], b, v) == 0.0
    @test extract([prevfloat(0.95)], b, v) == 1.0
    @test extract([nextfloat(1.05)], b, v) == 1.0
end

@testitem "Infinite bounds give a one-sided limit" setup=[Backends] tags=[:unit, :fast] begin
    b = VecBackend()
    upper = ViolationEvent(VoltageMagnitude(1), -Inf, 1.05)
    lower = ViolationEvent(VoltageMagnitude(1), 0.95, Inf)

    @test extract([0.5], b, upper) == 0.0
    @test extract([1.1], b, upper) == 1.0
    @test extract([1.5], b, lower) == 0.0
    @test extract([0.9], b, lower) == 1.0
end

@testitem "A ViolationEvent needs no method from the backend" setup=[Backends] tags=[
    :unit,
    :fast,
] begin
    v = ViolationEvent(VoltageMagnitude(2), 0.95, 1.05)

    @test extract([1.0, 1.10], VecBackend(), v) == 1.0
    @test extract(Dict(1 => 1.0, 2 => 1.10), DictBackend(), v) == 1.0
end

@testitem "extract on a ViolationEvent is type stable" setup=[Backends] tags=[:unit, :fast] begin
    b = VecBackend()
    v = ViolationEvent(VoltageMagnitude(2), 0.95, 1.05)

    @test @inferred(extract([1.0, 1.10], b, v)) isa Float64
end

@testitem "A NaN reading counts as a violation" setup=[Backends] tags=[:unit, :fast] begin
    b = VecBackend()
    v = ViolationEvent(VoltageMagnitude(1), 0.95, 1.05)

    @test extract([NaN], b, v) == 1.0
end
