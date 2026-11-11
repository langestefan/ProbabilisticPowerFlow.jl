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
