@testitem "ComponentRef is an isbits reference" tags=[:unit, :fast] begin
    r = ComponentRef(ComponentField.Pd, 3)
    @test isbitstype(ComponentRef)
    @test r.field === ComponentField.Pd
    @test r.id == 3
    @test r == ComponentRef(ComponentField.Pd, 3)
end

@testitem "kind covers every ComponentField member" tags=[:unit, :fast] begin
    for f in instances(ComponentField.T)
        @test kind(f) isa ComponentKind.T
    end

    @test kind(ComponentField.Pd) === ComponentKind.Load
    @test kind(ComponentField.Qd) === ComponentKind.Load
    @test kind(ComponentField.Pg) === ComponentKind.Gen
    @test kind(ComponentField.Qg) === ComponentKind.Gen
    @test kind(ComponentField.Vg) === ComponentKind.Gen
    @test kind(ComponentField.Vm) === ComponentKind.Bus
end

@testitem "kind delegates from a ComponentRef to its field" tags=[:unit, :fast] begin
    @test kind(ComponentRef(ComponentField.Vm, 1)) === ComponentKind.Bus
    @test kind(ComponentRef(ComponentField.Pg, 2)) === ComponentKind.Gen
end

@testitem "SolveInfo records the outcome of one solve" tags=[:unit, :fast] begin
    ok = SolveInfo(true, 4, 1e-10)
    @test ok.converged
    @test ok.iterations == 4
    @test ok.residual == 1e-10

    # A diverged solve whose solver reports neither iterations nor residual.
    diverged = SolveInfo(false, -1, Inf)
    @test !diverged.converged
    @test diverged.iterations == -1
    @test isinf(diverged.residual)
end

@testitem "supports_warmstart defaults to false" tags=[:unit, :fast] begin
    struct OptOutBackend <: AbstractPFBackend end
    @test supports_warmstart(OptOutBackend()) === false
end
