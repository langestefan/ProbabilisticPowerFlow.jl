module PPFNonlinearSolveExt

import NonlinearSolve as NLS
import PowerModels as PM
import ProbabilisticPowerFlow as PPF
import ProbabilisticPowerFlow: SolveInfo
using LinearAlgebra: norm

const NLSAlgorithm = NLS.SciMLBase.AbstractNonlinearAlgorithm

PPF.supports_pf_solver(::NLSAlgorithm) = true

# The residual and Jacobian below port the mismatch equations of
# PowerModels._compute_ac_pf onto the same PowerFlowData arrays. One deliberate
# difference: both callbacks update the operating point from x themselves, so
# neither depends on the other having run at the same point.
function update_operating_point!(pf, x)
    bt = pf.bus_type_idx
    @inbounds for i in eachindex(pf.am.idx_to_bus)
        if bt[i] == 1
            pf.vm_idx[i] = x[2i-1]
            pf.va_idx[i] = x[2i]
        elseif bt[i] == 2
            pf.q_inject_idx[i] = x[2i-1]
            pf.va_idx[i] = x[2i]
        else
            pf.p_inject_idx[i] = x[2i-1]
            pf.q_inject_idx[i] = x[2i]
        end
    end
    return nothing
end

function make_residual(pf)
    return function (F, x, p)
        update_operating_point!(pf, x)
        am = pf.am
        vm = pf.vm_idx
        va = pf.va_idx
        @inbounds for i in eachindex(am.idx_to_bus)
            balance_real = pf.p_delta_base_idx[i] + pf.p_inject_idx[i]
            balance_imag = pf.q_delta_base_idx[i] + pf.q_inject_idx[i]
            for j in pf.neighbors[i]
                if i == j
                    balance_real += vm[i] * vm[i] * real(am.matrix[i, i])
                    balance_imag += vm[i] * vm[i] * -imag(am.matrix[i, i])
                else
                    y = am.matrix[i, j]
                    c = cos(va[i] - va[j])
                    s = sin(va[i] - va[j])
                    balance_real += vm[i] * vm[j] * (real(y) * c + imag(y) * s)
                    balance_imag += vm[i] * vm[j] * (-imag(y) * c + real(y) * s)
                end
            end
            F[2i-1] = balance_real
            F[2i] = balance_imag
        end
        return nothing
    end
end

function make_jacobian(pf)
    return function (J, x, p)
        update_operating_point!(pf, x)
        am = pf.am
        bt = pf.bus_type_idx
        vm = pf.vm_idx
        va = pf.va_idx
        @inbounds for i in eachindex(am.idx_to_bus)
            f_r = 2i - 1
            f_i = 2i
            for j in pf.neighbors[i]
                x1 = 2j - 1
                x2 = 2j
                if bt[j] == 1
                    if i == j
                        sr = 0.0
                        si = 0.0
                        tr = 0.0
                        ti = 0.0
                        for k in pf.neighbors[i]
                            k == i && continue
                            y = am.matrix[i, k]
                            c = cos(va[i] - va[k])
                            s = sin(va[i] - va[k])
                            sr += real(y) * vm[k] * c + imag(y) * vm[k] * s
                            si += -imag(y) * vm[k] * c + real(y) * vm[k] * s
                            tr += real(y) * vm[k] * -s + imag(y) * vm[k] * c
                            ti += -imag(y) * vm[k] * -s + real(y) * vm[k] * c
                        end
                        y_ii = am.matrix[i, i]
                        J[f_r, x1] = 2 * real(y_ii) * vm[i] + sr
                        J[f_r, x2] = vm[i] * tr
                        J[f_i, x1] = -2 * imag(y_ii) * vm[i] + si
                        J[f_i, x2] = vm[i] * ti
                    else
                        y = am.matrix[i, j]
                        c = cos(va[i] - va[j])
                        s = sin(va[i] - va[j])
                        J[f_r, x1] = vm[i] * (real(y) * c + imag(y) * s)
                        J[f_r, x2] = vm[i] * vm[j] * (real(y) * s - imag(y) * c)
                        J[f_i, x1] = vm[i] * (-imag(y) * c + real(y) * s)
                        J[f_i, x2] = vm[i] * vm[j] * (-imag(y) * s - real(y) * c)
                    end
                elseif bt[j] == 2
                    if i == j
                        tr = 0.0
                        ti = 0.0
                        for k in pf.neighbors[i]
                            k == i && continue
                            y = am.matrix[i, k]
                            c = cos(va[i] - va[k])
                            s = sin(va[i] - va[k])
                            tr += real(y) * vm[k] * -s + imag(y) * vm[k] * c
                            ti += -imag(y) * vm[k] * -s + real(y) * vm[k] * c
                        end
                        J[f_r, x1] = 0.0
                        J[f_i, x1] = 1.0
                        J[f_r, x2] = vm[i] * tr
                        J[f_i, x2] = vm[i] * ti
                    else
                        y = am.matrix[i, j]
                        c = cos(va[i] - va[j])
                        s = sin(va[i] - va[j])
                        J[f_r, x1] = 0.0
                        J[f_i, x1] = 0.0
                        J[f_r, x2] = vm[i] * vm[j] * (real(y) * s - imag(y) * c)
                        J[f_i, x2] = vm[i] * vm[j] * (-imag(y) * s - real(y) * c)
                    end
                else
                    if i == j
                        J[f_r, x1] = 1.0
                        J[f_r, x2] = 0.0
                        J[f_i, x1] = 0.0
                        J[f_i, x2] = 1.0
                    end
                end
            end
        end
        return nothing
    end
end

# Flat start replicates the PowerModels convention: PQ magnitudes at 1, every
# other unknown at 0. A warm start from a solved state reuses its solution
# vector directly. A warm start from an unsolved state reads vm and va from its
# data dictionary, which supports seeding from dataset starting points.
function build_x0!(x0, pf, warmstart)
    fill!(x0, 0.0)
    bt = pf.bus_type_idx
    if warmstart === nothing
        @inbounds for i in eachindex(bt)
            bt[i] == 1 && (x0[2i-1] = 1.0)
        end
    elseif warmstart.has_last
        copyto!(x0, warmstart.last_x)
    else
        data = warmstart.data
        @inbounds for (i, bid) in enumerate(pf.am.idx_to_bus)
            bus = data["bus"]["$(bid)"]
            if bt[i] == 1
                x0[2i-1] = bus["vm"]::Float64
                x0[2i] = bus["va"]::Float64
            elseif bt[i] == 2
                x0[2i] = bus["va"]::Float64
            end
        end
    end
    return x0
end

# The NonlinearSolve solver path: the same equations as the bundled path, on a
# nonlinear cache that is built once per state and reused, so repeated solves
# allocate almost nothing.
function PPF.run_pf_solver!(state, b, alg::NLSAlgorithm, warmstart)
    pf = state.pf_data
    x0 = Vector{Float64}(undef, 2 * length(pf.am.idx_to_bus))
    build_x0!(x0, pf, warmstart)
    if state.solver_cache === nothing
        nf = NLS.NonlinearFunction(
            make_residual(pf);
            jac = make_jacobian(pf),
            jac_prototype = copy(pf.J0),
        )
        prob = NLS.NonlinearProblem(nf, x0, nothing)
        state.solver_cache = NLS.init(prob, alg; abstol = b.tol, maxiters = b.maxiter)
    else
        # p must be passed explicitly: without it reinit! rebuilds its
        # parameters with Missing where the cache was built with nothing,
        # which trips a TypeError inside NonlinearSolveBase
        NLS.reinit!(state.solver_cache, x0; p = nothing)
    end
    sol = NLS.solve!(state.solver_cache)
    converged = sol.retcode == NLS.ReturnCode.Success
    if converged
        resize!(state.last_x, length(sol.u))
        copyto!(state.last_x, sol.u)
        state.has_last = true
    end
    return SolveInfo(converged, sol.stats.nsteps, norm(sol.resid, Inf)), sol.u
end

end
