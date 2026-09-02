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

# 61 buses at 110 kV plus the three EHV boundary nodes the grid hangs from, so the
# voltage level doubles as a map of where this network connects upward.
for bus in values(data["bus"])
    bus["level"] = "$(round(Int, bus["base_kv"])) kV"
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
    # Vega orders a nominal scale alphabetically, and "110 kV" < "220 kV" < "380 kV"
    # sorts the way the ladder does.
    bus = (
        :data => :level,
        :data_type => :nominal,
        :size => 45,
        :color => ["#0072B2", "#009E73", "#CC79A7"],
    ),
    branch = (:color => "#000000", :size => 1.1),
    width = 470,
    height = 380,
)

# The two farms sit 0.4 km apart, which on this map is five pixels, so a ring and a
# label do the pointing that two adjacent markers cannot.
spot = data["bus"]["$(OBSERVED)"]
ring = @vlplot(
    data = (values = [(x = spot["xcoord_1"], y = spot["ycoord_1"])],),
    mark = {
        :point,
        shape = "circle",
        size = 420,
        filled = false,
        strokeWidth = 1.8,
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
            t = "wind farms $(OTHER) & $(OBSERVED), voltage at $(OBSERVED)",
        )],
    ),
    mark =
        {:text, dx = -28, dy = -20, align = "right", fontSize = 14, color = "#D55E00"},
    x = {"x:q", axis = nothing, scale = {zero = false}},
    y = {"y:q", axis = nothing, scale = {zero = false}},
    text = {"t:n"},
)

# A transformer's two buses share a substation and therefore a coordinate, so the one
# 220 kV and two 380 kV nodes are drawn underneath their 110 kV twins and vanish. They
# are redrawn on top, larger, sharing the bus colour scale so no second legend appears.
const LEVEL_SCALE =
    (domain = ["110 kV", "220 kV", "380 kV"], range = ["#0072B2", "#009E73", "#CC79A7"])

ehv = [
    (x = b["xcoord_1"], y = b["ycoord_1"], level = b["level"]) for
    b in values(data["bus"]) if b["base_kv"] > 150
]
upper = @vlplot(
    data = (values = ehv,),
    mark = {:point, shape = "circle", size = 110, filled = true, opacity = 1.0},
    x = {"x:q", axis = nothing, scale = {zero = false}},
    y = {"y:q", axis = nothing, scale = {zero = false}},
    color = {"level:n", scale = LEVEL_SCALE, legend = nothing},
)

out = joinpath(FIGDIR, "network.png")
save(out, p + upper + ring + label)
println("Wrote ", relpath(out, pwd()))
