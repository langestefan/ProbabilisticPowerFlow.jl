@testitem "Aqua quality assurance" tags = [:quality] begin
    using Aqua: Aqua

    # persistent_tasks precompiles a wrapper package in a subprocess and waits for
    # it to report back. On the Windows runner that subprocess exits without
    # reporting, which fails the check for reasons that have nothing to do with
    # this package: it starts no tasks at load, and the check passes everywhere
    # else.
    Aqua.test_all(ProbabilisticPowerFlow; persistent_tasks = !Sys.iswindows())
end

@testitem "JET static analysis" tags = [:quality] begin
    # JET tracks the compiler internals of one Julia version at a time, so the
    # analysis only runs on the versions this package has verified it against.
    if v"1.12" <= VERSION < v"1.13"
        using JET: JET

        # target_modules keeps the report on code this package owns. Without it
        # the analysis of an abstractly typed argument walks into every
        # implementation that could satisfy it, which reports on Base and on
        # packages this one only calls into.
        JET.test_package(ProbabilisticPowerFlow; target_modules = (ProbabilisticPowerFlow,))
    end
end
