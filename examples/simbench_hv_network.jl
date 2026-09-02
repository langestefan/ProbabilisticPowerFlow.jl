# The topology of the SimBench HV grid, with the two wind farms the correlation study
# varies and the bus whose voltage it watches marked on it.
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
const OTHER = 54             # the second farm, whose correlation with it is the parameter
const FIGDIR = joinpath(@__DIR__, "figures", "simbench_hv")
mkpath(FIGDIR)

grid = read_grid(GRID)
data = powermodels_data(grid)

# SimBench ships real coordinates, so the plot is the network as it lies on the ground
# rather than a layout algorithm's guess.
SimBench.attach_coordinates!(data, grid)

# Topology only. The loads and generators would add 164 more markers and say nothing
# about where the study happens, so they are dropped rather than drawn and ignored.
for table in ("load", "gen", "shunt", "storage")
    haskey(data, table) && empty!(data[table])
end

for bus in values(data["bus"])
    i = bus["index"]
    bus["role"] = if i == OBSERVED
        "wind farm $(OBSERVED), voltage watched"
    elseif i == OTHER
        "wind farm $(OTHER)"
    else
        "bus"
    end
end

println("$(GRID): $(length(data["bus"])) buses, $(length(data["branch"])) branches")

# ---------------------------------------------------------------------------
# The plot.
#
# PowerPlots offsets parallel circuits so both are visible, by 0.05 of a coordinate
# unit. attach_coordinates! normalises the map into a unit box, so that default shifts
# a branch by a tenth of the whole grid, and 70 of these 101 branches run as double
# circuits and would be drawn nowhere near their buses.
# ---------------------------------------------------------------------------

p = powerplot(
    data;
    fixed = true,
    parallel_edge_offset = 0.002,
    # Vega orders a nominal scale alphabetically: bus, wind farm 54, wind farm 61.
    bus = (
        :data => :role,
        :data_type => :nominal,
        :size => 80,
        :color => ["#aab4c0", "#0072B2", "#D55E00"],
    ),
    branch = (:color => "#7d8896", :size => 2),
    width = 900,
    height = 720,
)

# The two farms sit 0.4 km apart, which on this map is five pixels, so a ring and a
# label do the pointing that two adjacent markers cannot.
spot = data["bus"]["$(OBSERVED)"]
ring = @vlplot(
    data = (values = [(x = spot["xcoord_1"], y = spot["ycoord_1"])],),
    mark = {
        :point,
        shape = "circle",
        size = 1500,
        filled = false,
        strokeWidth = 2.5,
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
            t = "wind farms $(OTHER) & $(OBSERVED)",
        )],
    ),
    mark =
        {:text, dx = -28, dy = -20, align = "right", fontSize = 14, color = "#D55E00"},
    x = {"x:q", axis = nothing, scale = {zero = false}},
    y = {"y:q", axis = nothing, scale = {zero = false}},
    text = {"t:n"},
)

out = joinpath(FIGDIR, "network.png")
save(out, p + ring + label)
println("Wrote ", relpath(out, pwd()))
