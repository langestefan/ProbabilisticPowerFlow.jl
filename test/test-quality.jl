@testitem "Aqua quality assurance" tags = [:quality] begin
    using Aqua: Aqua
    Aqua.test_all(ProbabilisticPowerFlow)
end

@testitem "JET static analysis" tags = [:quality] begin
    using JET: JET
    JET.test_package(ProbabilisticPowerFlow; target_modules = (ProbabilisticPowerFlow,))
end
