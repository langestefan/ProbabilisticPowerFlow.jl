# Figures for the SimBench HV study.
#
#   julia --project=examples -t auto examples/simbench_hv_plots.jl
#
# Runs examples/simbench_hv.jl and draws what its numbers look like. Kept separate so
# the study itself does not depend on Makie, which is by far the heaviest thing in the
# examples environment.

include(joinpath(@__DIR__, "simbench_hv.jl"))

using CairoMakie
using KernelDensity: kde
using Statistics: median

const FIGDIR = joinpath(@__DIR__, "figures", "simbench_hv")
mkpath(FIGDIR)

# Colour-blind safe, and consistent across every figure: one colour for demand, one
# for wind, one for everything the network does in response.
const C_LOAD = "#0072B2"
const C_WIND = "#009E73"
const C_NET = "#D55E00"
const C_GREY = "#666666"

set_theme!(
    Theme(
        fontsize = 15,
        Axis = (
            xgridvisible = false,
            ygridcolor = (:black, 0.06),
            topspinevisible = false,
            rightspinevisible = false,
        ),
    ),
)

save_fig(name, fig) = (save(joinpath(FIGDIR, name), fig; px_per_unit = 2); name)
saved = String[]

# The samples are stored in solve order, which :sorted permutes away from draw order.
# Anything that reads the samples as a sequence has to undo that first.
draw_order = sortperm(result.sample_indices)

vm = Dict(b => qoi_samples(result, VoltageMagnitude(b)) for b in buses)

# ---------------------------------------------------------------------------
# 1. Every bus, sorted by median voltage. The p5-p95 band is the sampling answer;
#    the min-max whisker is what 5000 draws actually reached.
# ---------------------------------------------------------------------------

let
    order = sort(buses; by = b -> median(vm[b]))
    xs = 1:length(order)
    p05 = [quantile(vm[b], 0.05) for b in order]
    p50 = [median(vm[b]) for b in order]
    p95 = [quantile(vm[b], 0.95) for b in order]
    lo = [minimum(vm[b]) for b in order]
    hi = [maximum(vm[b]) for b in order]

    lo_all = minimum(lo)
    hi_all = maximum(hi)

    fig = Figure(size = (1000, 470))
    ax = Axis(
        fig[1, 1],
        title = "Bus voltage across $(n_converged(result)) samples, $(GRID)",
        xlabel = "bus, ordered by median voltage",
        ylabel = "voltage magnitude (pu)",
    )
    rangebars!(ax, xs, lo, hi; color = (C_GREY, 0.45), linewidth = 1)
    band!(ax, xs, p05, p95; color = (C_LOAD, 0.35))
    scatter!(ax, xs, p50; color = C_LOAD, markersize = 5)
    ylims!(ax, lo_all - 0.002, hi_all + 0.006)

    elems = [
        PolyElement(color = (C_LOAD, 0.35)),
        LineElement(color = C_GREY),
        MarkerElement(color = C_LOAD, marker = :circle, markersize = 8),
    ]
    axislegend(
        ax,
        elems,
        ["p5 - p95", "min - max", "median"];
        position = :lt,
        framevisible = false,
        patchsize = (18, 12),
    )

    # The limits are so far away that drawing them on the same axis would flatten the
    # spread to nothing. This panel is the same data against the full band instead, and
    # is the honest answer to "how close did it get".
    ax2 = Axis(fig[1, 2], title = "vs limits", yaxisposition = :right)
    band!(ax2, [0.0, 1.0], fill(lo_all, 2), fill(hi_all, 2); color = (C_LOAD, 0.55))
    hlines!(ax2, [LO, HI]; color = C_NET, linestyle = :dash, linewidth = 2)
    ylims!(ax2, LO - 0.02, HI + 0.02)
    xlims!(ax2, 0, 1)
    hidexdecorations!(ax2)
    text!(
        ax2,
        0.5,
        HI - 0.004;
        text = "$(round(HI - hi_all, digits = 3)) spare",
        align = (:center, :top),
        fontsize = 12,
        color = C_GREY,
    )
    colsize!(fig.layout, 2, Relative(0.13))

    push!(saved, save_fig("bus-voltage-bands.png", fig))
end

# ---------------------------------------------------------------------------
# 2. The marginals, which are the SimBench profiles themselves. This is the figure
#    that explains why they are empirical: wind spends most of the year near zero
#    and occasionally sits at rated output, and no standard family does that while
#    also describing the tight, unimodal load profiles beside it.
# ---------------------------------------------------------------------------

let
    fig = Figure(size = (980, 400))
    winds = [v for v in model.variables if startswith(v.id, "res:WP")]
    loads =
        [v for v in model.variables if endswith(v.id, ":p") && startswith(v.id, "load:")]

    ax1 = Axis(
        fig[1, 1],
        title = "Wind profiles",
        xlabel = "output / rating",
        ylabel = "density",
    )
    for (k, v) in enumerate(winds)
        density!(
            ax1,
            v.dist.support;
            weights = v.dist.p,
            color = (C_WIND, 0.12),
            strokecolor = C_WIND,
            strokewidth = 1.6,
            strokearound = false,
            label = replace(v.id, "res:" => ""),
        )
    end
    axislegend(ax1; position = :rt, framevisible = false)

    ax2 =
        Axis(fig[1, 2], title = "Load profiles (active power)", xlabel = "demand / rating")
    for v in loads
        density!(
            ax2,
            v.dist.support;
            weights = v.dist.p,
            color = (C_LOAD, 0.10),
            strokecolor = C_LOAD,
            strokewidth = 1.6,
            strokearound = false,
            label = replace(v.id, "load:" => "", ":p" => ""),
        )
    end
    axislegend(ax2; position = :rt, framevisible = false)
    linkxaxes!(ax1, ax2)
    xlims!(ax1, -0.02, 1.02)

    Label(
        fig[0, :],
        "Germ marginals, taken from $(size(Z, 1)) quarter-hourly SimBench steps",
        fontsize = 17,
        padding = (0, 0, 4, 0),
    )
    push!(saved, save_fig("germ-marginals.png", fig))
end

# ---------------------------------------------------------------------------
# 3. The copula's correlation matrix. The block structure is the point: load p and q
#    of the same profile hang together, the three wind parks hang together, and the
#    two blocks are mildly opposed.
# ---------------------------------------------------------------------------

let
    labels = [v.id for v in model.variables]
    d = length(labels)
    fig = Figure(size = (760, 660))
    ax = Axis(
        fig[1, 1],
        title = "Gaussian copula correlation, estimated from the profiles",
        xticks = (1:d, labels),
        yticks = (1:d, labels),
        xticklabelrotation = π / 3,
        xticklabelsize = 11,
        yticklabelsize = 11,
        yreversed = true,
    )
    hm = heatmap!(ax, 1:d, 1:d, R; colormap = :RdBu, colorrange = (-1, 1))
    Colorbar(fig[1, 2], hm; label = "correlation")
    push!(saved, save_fig("copula-correlation.png", fig))
end

# ---------------------------------------------------------------------------
# 4. Line loading, for the circuits a bus pair can address. The gap to 1.0 is the
#    answer a planner wants when nothing violates.
# ---------------------------------------------------------------------------

let
    top = first(loading, 12)
    cats = Int[]
    vals = Float64[]
    for (k, (_, x)) in enumerate(top)
        append!(cats, fill(k, length(x)))
        append!(vals, x)
    end

    fig = Figure(size = (900, 470))
    ax = Axis(
        fig[1, 1],
        title = "Line loading on $(length(addressable)) of " *
                "$(count(b -> b["br_status"] != 0, values(data["branch"]))) branches; " *
                "the rest share a bus pair",
        xlabel = "|S| / rate_a",
        yticks = (1:length(top), ["$(f) → $(t)" for ((f, t), _) in top]),
        ylabel = "circuit",
        yreversed = true,
    )
    boxplot!(
        ax,
        cats,
        vals;
        orientation = :horizontal,
        color = (C_NET, 0.45),
        mediancolor = C_NET,
        whiskerwidth = 0.5,
        markersize = 3,
        outliercolor = (C_NET, 0.25),
    )
    vlines!(ax, [1.0]; color = :black, linestyle = :dash, linewidth = 2)
    text!(ax, 0.99, 0.65; text = "thermal rating ", align = (:right, :top), fontsize = 13)
    xlims!(ax, 0, 1.04)
    push!(saved, save_fig("line-loading.png", fig))
end

# ---------------------------------------------------------------------------
# 5. The mechanism. Voltage on this grid is a wind story: 1077 MW of wind capacity
#    against 522 MW of load, so the highest voltages are the windy hours.
# ---------------------------------------------------------------------------

let
    wind_j =
        [j for (j, a) in enumerate(model.assignments) if startswith(a.variable, "res:WP")]
    load_j = [
        j for (j, a) in enumerate(model.assignments) if
        startswith(a.variable, "load:") && a.target.field === ComponentField.Pd
    ]

    nsamp = n_converged(result)
    wind_mw = Vector{Float64}(undef, nsamp)
    load_mw = Vector{Float64}(undef, nsamp)
    for i = 1:nsamp
        x = to_physical(model, view(result.u, :, i))
        # RES enter as negative loads, so their injection is the negated sum
        wind_mw[i] = -sum(x[j] for j in wind_j) * data["baseMVA"]
        load_mw[i] = sum(x[j] for j in load_j) * data["baseMVA"]
    end
    vmax = [maximum(vm[b][i] for b in buses) for i = 1:nsamp]

    fig = Figure(size = (940, 470))
    ax = Axis(
        fig[1, 1],
        title = "Highest bus voltage against wind infeed",
        xlabel = "wind infeed (MW)",
        ylabel = "max bus voltage (pu)",
    )
    sc = scatter!(
        ax,
        wind_mw,
        vmax;
        color = load_mw,
        colormap = :viridis,
        markersize = 4,
        alpha = 0.55,
    )
    Colorbar(fig[1, 2], sc; label = "demand (MW)")
    push!(saved, save_fig("wind-vs-voltage.png", fig))
end

# ---------------------------------------------------------------------------
# 6. Convergence. The estimate wanders inside a band that closes as 1/sqrt(n), which
#    is the reason plain Monte Carlo needs many samples for a rare event and few for
#    a mean.
# ---------------------------------------------------------------------------

let
    b = argmax(bus -> maximum(vm[bus]), buses)
    x = vm[b][draw_order]           # back into draw order
    ns = 1:length(x)
    running = cumsum(x) ./ ns
    sd = [k < 2 ? 0.0 : std(view(x, 1:k)) for k in ns]
    se = 1.96 .* sd ./ sqrt.(ns)

    fig = Figure(size = (900, 430))
    ax = Axis(
        fig[1, 1],
        title = "Running estimate of mean voltage at bus $(b), with 95% interval",
        xlabel = "samples drawn",
        ylabel = "mean voltage (pu)",
    )
    band!(ax, ns, running .- se, running .+ se; color = (C_LOAD, 0.25))
    lines!(ax, ns, running; color = C_LOAD, linewidth = 1.6)
    hlines!(ax, [running[end]]; color = C_GREY, linestyle = :dash)
    vis = 20:length(x)
    pad = 0.12 * (maximum(running[vis] .+ se[vis]) - minimum(running[vis] .- se[vis]))
    xlims!(ax, 20, length(x))
    ylims!(
        ax,
        minimum(running[vis] .- se[vis]) - pad,
        maximum(running[vis] .+ se[vis]) + pad,
    )
    push!(saved, save_fig("mc-convergence.png", fig))
end


# ---------------------------------------------------------------------------
# 7. The joint law of voltage and angle at one bus, as a function of how correlated
#    two neighbouring wind farms are, uprated until the band is actually at risk.
#
#    The study's germ carries one variable per profile, so the fourteen WP7 farms move
#    as one and their mutual correlation is not a parameter at all. To ask the question
#    the two farms in question are split out of that group and given a germ variable
#    each, sharing WP7's marginal. Their dependence is the one-factor construction
#
#        z = sqrt(rho) * (regional WP7 factor) + sqrt(1 - rho) * (local noise)
#
#    which makes corr(A, B) = rho exactly, keeps each farm's correlation with the rest
#    of the system at sqrt(rho) times WP7's own, and is positive definite by
#    construction for every rho < 1. rho is the spatial correlation of wind between the
#    two sites: 0 is independent weather at each mast, 1 is one wind field.
#
#    Bus 61 is a radial stub hanging off bus 54 by a single 130 MVA line, and both
#    farms inject pure active power as SimBench ships them. That combination has a
#    voltage ceiling: sweeping their output deterministically, bus 61 peaks at 1.028 pu
#    around 400 MW and then *falls*, because the reactive loss on the spur grows as the
#    square of the current while the resistive rise is only linear. Past about 1100 MW
#    there is no solution at all. No amount of extra megawatts reaches 1.10.
#
#    Crossing the band therefore takes reactive power, not more of it. The farms below
#    run at 0.93 leading, which is inside what a modern grid code asks a wind farm to
#    provide, and that is what puts the limit within reach.
# ---------------------------------------------------------------------------

let
    PROFILE = "res:WP7"
    BUS_A, BUS_B = 54, 61   # 0.4 km apart, and the pair this grid's voltages feel most
    OBSERVED = 61           # carries the larger of the two farms
    RHOS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    CLOUDS = [0.0, 0.4, 0.8, 0.95]   # the four drawn as densities

    SCALE = 6.0     # both farms uprated by this factor
    TANPHI = 0.4    # reactive injection per unit of active, i.e. 0.93 leading

    farm(bus) = first(
        parse(Int, i) for (i, l) in data["load"] if l["source_id"][1] == "sgen" &&
            l["load_bus"] == bus &&
            profile_of[l["name"]] == PROFILE
    )
    ja, jb = farm(BUS_A), farm(BUS_B)

    w = findfirst(v -> v.id == PROFILE, model.variables)
    regional = model.variables[w].dist
    d0 = length(model.variables)

    variables2 = vcat(
        model.variables,
        [
            GermVariable("wind:bus$(BUS_A)", regional),
            GermVariable("wind:bus$(BUS_B)", regional),
        ],
    )

    # Uprating a farm is a change to its transform, not to its germ: the profile it
    # follows keeps its shape, the megawatts that shape turns into are larger. The
    # reactive injection is a second assignment onto the same germ variable, which is
    # the constant power factor pattern and keeps q locked to p sample by sample.
    uprate(t::AffineTransform) = AffineTransform(SCALE * t.a, t.b)

    assignments2 = Assignment[]
    for a in model.assignments
        if a.target.id != ja && a.target.id != jb
            push!(assignments2, a)
            continue
        end
        v = a.target.id == ja ? "wind:bus$(BUS_A)" : "wind:bus$(BUS_B)"
        push!(assignments2, Assignment(v, a.target, uprate(a.transform)))
        a.target.field === ComponentField.Pd || continue
        push!(
            assignments2,
            Assignment(
                v,
                ComponentRef(ComponentField.Qd, a.target.id),
                AffineTransform(TANPHI * SCALE * a.transform.a, 0.0),
            ),
        )
    end

    rated(j) = -SCALE * data["load"]["$(j)"]["pd"] * data["baseMVA"]
    @printf(
        "\nWind farms uprated %.0fx at %.2f leading: bus %d now %.0f MW, bus %d now %.0f MW\n",
        SCALE,
        cos(atan(TANPHI)),
        BUS_A,
        rated(ja),
        BUS_B,
        rated(jb),
    )

    function extended_R(rho)
        M = zeros(d0 + 2, d0 + 2)
        M[1:d0, 1:d0] .= R
        s = sqrt(rho)
        for k = 1:d0
            M[d0+1, k] = M[k, d0+1] = s * R[w, k]
            M[d0+2, k] = M[k, d0+2] = s * R[w, k]
        end
        M[d0+1, d0+2] = M[d0+2, d0+1] = rho
        M[d0+1, d0+1] = M[d0+2, d0+2] = 1.0
        return M
    end

    qois2 = AbstractQoI[VoltageMagnitude(OBSERVED), VoltageAngle(OBSERVED)]
    runs = map(RHOS) do rho
        M = extended_R(rho)
        @assert isposdef(M)
        m = UncertaintyModel(variables2, assignments2, GaussianCopula(M))
        # the same seed every time, so the clouds differ because rho differs and not
        # because the draws did
        r = solve(
            PPFProblem(backend, m, qois2),
            MonteCarlo(n = 4000, warmstart = :sorted);
            rng = Xoshiro(20260902),
            ntasks = Threads.nthreads(),
        )
        va = collect(qoi_samples(r, VoltageAngle(OBSERVED)))
        vmg = collect(qoi_samples(r, VoltageMagnitude(OBSERVED)))
        pv =
            violation_probability(r, ViolationEvent(VoltageMagnitude(OBSERVED), LO, HI))
        return (; rho, va, vmg, pv, n = n_converged(r), fails = length(r.failures))
    end

    # The level enclosing a given probability mass, read off the density itself rather
    # than assumed from a fitted shape.
    function hdr_level(dens, prob)
        z = sort(vec(dens); rev = true)
        c = cumsum(z)
        return z[findfirst(>=(prob * c[end]), c)]
    end

    kdes = [kde((r.va, r.vmg)) for r in runs]
    xl = extrema(vcat((r.va for r in runs)...))
    yl = extrema(vcat((r.vmg for r in runs)...))
    pad(t, f) = (t[1] - f * (t[2] - t[1]), t[2] + f * (t[2] - t[1]))
    xl, yl = pad(xl, 0.05), pad(yl, 0.05)
    colour(k) = cgrad(:viridis)[(k-1)/(length(runs)-1)]

    fig = Figure(size = (1120, 830))
    for (col, rho) in enumerate(CLOUDS)
        k = findfirst(==(rho), RHOS)
        r, kd = runs[k], kdes[k]
        ax = Axis(
            fig[1, col],
            title = "ρ = $(rho),  P(>$(HI)) = $(round(100 * r.pv, digits = 2))%",
            titlesize = 13,
            xlabel = "voltage angle (rad)",
            ylabel = col == 1 ? "voltage magnitude (pu)" : "",
            xticklabelsize = 11,
        )
        heatmap!(ax, kd.x, kd.y, kd.density; colormap = :dense)
        contour!(
            ax,
            kd.x,
            kd.y,
            kd.density;
            levels = [hdr_level(kd.density, 0.9), hdr_level(kd.density, 0.5)],
            color = :white,
            linewidth = 1.4,
        )
        hlines!(ax, [HI]; color = C_NET, linestyle = :dash, linewidth = 2)
        xlims!(ax, xl)
        ylims!(ax, yl)
        col == 1 || hideydecorations!(ax; grid = false)
    end

    ax = Axis(
        fig[2, 1:4],
        title = "90% contours, every ρ",
        xlabel = "voltage angle at bus $(OBSERVED) (rad)",
        ylabel = "voltage magnitude (pu)",
    )
    for (k, (r, kd)) in enumerate(zip(runs, kdes))
        contour!(
            ax,
            kd.x,
            kd.y,
            kd.density;
            levels = [hdr_level(kd.density, 0.9)],
            color = colour(k),
            linewidth = 2.4,
        )
    end
    hlines!(ax, [HI]; color = C_NET, linestyle = :dash, linewidth = 2)
    text!(
        ax,
        xl[2],
        HI + 0.0008;
        text = "limit $(HI) ",
        color = C_NET,
        align = (:right, :bottom),
        fontsize = 13,
    )
    xlims!(ax, xl)
    ylims!(ax, yl)
    # contour! does not hand its colour to the legend, so the swatches are built here
    axislegend(
        ax,
        [LineElement(color = colour(k), linewidth = 3) for k in eachindex(runs)],
        ["ρ = $(r.rho)" for r in runs],
        position = :lt,
        framevisible = false,
        rowgap = 1,
        nbanks = 2,
        patchsize = (22, 10),
    )

    ax2 = Axis(
        fig[3, 1:2],
        title = "Spread, relative to independent farms",
        titlesize = 13,
        xlabel = "ρ",
        ylabel = "sd(ρ) / sd(0)",
    )
    sv = [std(r.vmg) for r in runs]
    sa = [std(r.va) for r in runs]
    scatterlines!(ax2, RHOS, sv ./ sv[1]; color = C_LOAD, linewidth = 2, label = "voltage")
    scatterlines!(ax2, RHOS, sa ./ sa[1]; color = C_NET, linewidth = 2, label = "angle")
    axislegend(ax2; position = :rb, framevisible = false)

    # The headline: the same hardware and the same weather, and the chance of leaving
    # the band is set by whether the two farms blow together.
    ax3 = Axis(
        fig[3, 3:4],
        title = "P(voltage above $(HI))",
        titlesize = 13,
        xlabel = "ρ",
        ylabel = "probability",
    )
    pv = [r.pv for r in runs]
    se = [1.96 * sqrt(p * (1 - p) / r.n) for (p, r) in zip(pv, runs)]
    band!(ax3, RHOS, pv .- se, pv .+ se; color = (C_NET, 0.2))
    scatterlines!(ax3, RHOS, pv; color = C_NET, linewidth = 2)

    Label(
        fig[0, :],
        "Voltage and angle at bus $(OBSERVED), against the correlation between the " *
        "wind farms at buses $(BUS_A) and $(BUS_B), both uprated " *
        "$(round(Int, SCALE))x at $(round(cos(atan(TANPHI)), digits = 2)) leading",
        fontsize = 17,
        padding = (0, 0, 4, 0),
    )
    rowsize!(fig.layout, 2, Relative(0.32))
    rowsize!(fig.layout, 3, Relative(0.22))
    push!(saved, save_fig("joint-voltage-angle.png", fig))

    println("\nBus $(OBSERVED), against the correlation between the two farms:")
    @printf("%6s %10s %10s %10s %12s\n", "ρ", "sd(vm)", "sd(va)", "max vm", "P(>$(HI))")
    for r in runs
        @printf(
            "%6.2f %10.5f %10.5f %10.4f %12.4f\n",
            r.rho,
            std(r.vmg),
            std(r.va),
            maximum(r.vmg),
            r.pv
        )
    end
end


println("\n", repeat("=", 72))
println("Wrote $(length(saved)) figures to $(relpath(FIGDIR, pwd()))")
for f in saved
    println("  ", f)
end
