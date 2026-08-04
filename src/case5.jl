"""
    case5(; load_scale = 1.0)

The Stagg & El-Abiad 5-bus test system (*Computer Methods in Power System Analysis*,
1968), with bus 2 promoted from PQ to PV so that the PV rows of the Jacobian are
exercised: bus 1 slack at 1.06 pu, bus 2 PV at 1.04 pu with 0.40 pu generation,
buses 3-5 PQ loads. All values per unit on a 100 MVA base.

`load_scale` multiplies every load; large values push the case past its loadability
limit, which tests genuinely exercise to produce diverged solves.
"""
function case5(; load_scale::Real = 1.0)
    branches = [
        Branch(1, 2, 0.02, 0.06, 0.06),
        Branch(1, 3, 0.08, 0.24, 0.05),
        Branch(2, 3, 0.06, 0.18, 0.04),
        Branch(2, 4, 0.06, 0.18, 0.04),
        Branch(2, 5, 0.04, 0.12, 0.03),
        Branch(3, 4, 0.01, 0.03, 0.02),
        Branch(4, 5, 0.08, 0.24, 0.05),
    ]
    bustype = [3, 2, 1, 1, 1]
    pd = load_scale .* [0.0, 0.20, 0.45, 0.40, 0.60]
    qd = load_scale .* [0.0, 0.10, 0.15, 0.05, 0.10]
    pg = [0.0, 0.40, 0.0, 0.0, 0.0]
    vm_setpoint = [1.06, 1.04, 1.0, 1.0, 1.0]
    return NetworkData(bustype, pd, qd, pg, vm_setpoint, branches)
end
