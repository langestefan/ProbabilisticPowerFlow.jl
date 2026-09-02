# The SimBench HV network, with the wind farms and the bus the correlation study
# watches marked on it.
#
#   julia --project=examples/network examples/simbench_hv_network.jl
#
# PowerPlots caps PowerModels at 0.21 and the PPF backend needs the 0.22 fork, so the
# two cannot share an environment. They do not have to: PowerPlots only ever reads the
# network data dictionary, and SimBench builds that without PowerModels at all. Hence
# examples/network/, which holds the registered PowerModels purely to draw with.

using SimBench
using PowerModels
using PowerPlots
using VegaLite

PowerModels.silence()

const GRID = "1-HV-mixed--0-no_sw"
const OBSERVED = 61          # the bus the joint voltage-angle study watches
const PAIR = (54, 61)        # the two farms whose correlation is the parameter
const FIGDIR = joinpath(@__DIR__, "figures", "simbench_hv")
mkpath(FIGDIR)

grid = read_grid(GRID)
data = powermodels_data(grid)

# Same operating point as the study: every study case puts the reference at 1.025,
# where the Node table's 1.092 belongs to the EHV grid these boundary nodes are cut from.
apply_study_case!(data, grid, "hW")

# SimBench ships real coordinates, so the plot is the network as it lies on the ground
# rather than a layout algorithm's guess.
SimBench.attach_coordinates!(data, grid)

# ---------------------------------------------------------------------------
# Marking. A SimBench RES unit arrives as a PowerModels load with a negative pd, so
# the wind farms have to be found through the RES table's profile column rather than
# by looking for generators.
# ---------------------------------------------------------------------------

profile_of = Dict(r.id => string(r.profile) for r in eachrow(grid[:RES]))
is_wind(name) = startswith(get(profile_of, name, ""), "WP")

wind_buses = Set{Int}()
for load in values(data["load"])
    load["source_id"][1] == "sgen" || continue
    is_wind(load["name"]) || continue
    load["pd"] == 0 && continue
    push!(wind_buses, load["load_bus"])
end

for load in values(data["load"])
    wind = load["source_id"][1] == "sgen" && is_wind(load["name"]) && load["pd"] != 0
    load["kind"] = wind ? "wind farm" : "demand / other DER"
end

# The two study buses lie 0.4 km apart, which at the scale of this map is one point:
# their normalised coordinates differ by 0.005 of a unit box. Marking them separately
# would just draw one marker on top of the other, so they share a category and the
# annotation below points at the spot.
for bus in values(data["bus"])
    i = bus["index"]
    bus["role"] = if i in PAIR
        "study pair $(first(PAIR)) & $(last(PAIR))"
    elseif i in wind_buses
        "wind farm bus"
    elseif bus["bus_type"] == 3
        "reference (EHV)"
    else
        "bus"
    end
end

# Branch loading at this operating point, so the drawing carries some electrical
# information rather than only topology.
result = compute_ac_pf(data)
@assert result["termination_status"]
update_data!(data, result["solution"])
update_data!(data, calc_branch_flow_ac(data))
for br in values(data["branch"])
    br["loading_pct"] = 100 * hypot(br["pf"], br["qf"]) / br["rate_a"]
end

println("$(GRID): $(length(data["bus"])) buses, $(length(data["branch"])) branches")
println("wind farm buses: $(length(wind_buses))")
println("observed bus $(OBSERVED), study pair $(PAIR)")

# ---------------------------------------------------------------------------
# The plot. Category order fixes the colour order, so the two marked roles get the
# strong colours and the ordinary buses stay grey.
# ---------------------------------------------------------------------------

# PowerPlots offsets parallel circuits so both are visible, by 0.05 of a coordinate
# unit. attach_coordinates! normalises the map into a unit box, so that default shifts
# a branch by a tenth of the whole grid and 70 of these 101 branches, which run as
# double circuits, end up drawn nowhere near their buses.
p = powerplot(
    data;
    fixed = true,
    parallel_edge_offset = 0.002,
    # Vega orders a nominal scale alphabetically, so the colours follow the category
    # names: bus, observed, reference, study pair, wind farm.
    bus = (
        :data => :role,
        :data_type => :nominal,
        :size => 170,
        :color => ["#9aa5b1", "#0072B2", "#D55E00", "#009E73"],
    ),
    # Branch loading runs light-to-dark rather than through a colour ramp, so that
    # every colour on the plot means a marked component and nothing else.
    branch = (
        :data => :loading_pct,
        :data_type => :quantitative,
        :color => ["#dfe4ea", "#1f2933"],
        :size => 3,
    ),
    load = (
        :data => :kind,
        :data_type => :nominal,
        :size => 35,
        :color => ["#c4cad3", "#009E73"],
    ),
    gen = (:size => 70, :color => "#0072B2"),
    connector = (:size => 0.7,),
    width = 920,
    height = 740,
)

# A ring and a label on the pair, layered onto the PowerPlots spec. The bus markers
# alone cannot say which of the 42 green dots the correlation study is about.
spot = data["bus"]["$(OBSERVED)"]
ring = @vlplot(
    data = (values = [(x = spot["xcoord_1"], y = spot["ycoord_1"])],),
    mark = {
        :point,
        shape = "circle",
        size = 1500,
        filled = false,
        strokeWidth = 3,
        stroke = "#D55E00",
    },
    x = {"x:q", axis = nothing, scale = {zero = false}},
    y = {"y:q", axis = nothing, scale = {zero = false}},
)
label = @vlplot(
    data = (
        values = [(
            x = spot["xcoord_1"],
            y = spot["ycoord_1"],
            t = "voltage watched here (bus $(OBSERVED))",
        )],
    ),
    mark =
        {:text, dx = -26, dy = -22, align = "right", fontSize = 13, color = "#D55E00"},
    x = {"x:q", axis = nothing, scale = {zero = false}},
    y = {"y:q", axis = nothing, scale = {zero = false}},
    text = {"t:n"},
)

out = joinpath(FIGDIR, "network.png")
save(out, p + ring + label)
println("\nWrote ", relpath(out, pwd()))
