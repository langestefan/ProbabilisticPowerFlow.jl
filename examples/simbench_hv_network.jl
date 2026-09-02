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

# SimBench ships real coordinates, but geography is not what makes a 64-bus network
# readable: on the map the two study farms land 0.4 km apart and the double circuits
# fold onto each other. The layout is computed from the topology instead, and written
# into the data so the annotations below can read the positions back out.
PowerPlots.layout_network!(data; layout_algorithm = kamada_kawai)

xs = [b["xcoord_1"] for b in values(data["bus"])]
ys = [b["ycoord_1"] for b in values(data["bus"])]
span = max(maximum(xs) - minimum(xs), maximum(ys) - minimum(ys))

println("$(GRID): $(length(data["bus"])) buses, $(length(data["branch"])) branches")
println("layout span $(round(span, digits = 2)) units")

# ---------------------------------------------------------------------------
# The plot. The parallel-circuit offset has to be a fraction of the layout's own
# coordinate span; 70 of these 101 branches run as double circuits, and an offset
# scaled for a unit box would put them nowhere near their buses.
# ---------------------------------------------------------------------------

p = powerplot(
    data;
    fixed = true,
    parallel_edge_offset = 0.006 * span,
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

# A transformer's two buses share a substation, so on the geographic map the one
# 220 kV and two 380 kV nodes hid underneath their 110 kV twins. The layout separates
# them, but they are still three markers among sixty-four, so they are redrawn larger,
# sharing the bus colour scale so no second legend appears.
const LEVEL_SCALE =
    (domain = ["110 kV", "220 kV", "380 kV"], range = ["#0072B2", "#009E73", "#CC79A7"])

ehv = [
    (x = b["xcoord_1"], y = b["ycoord_1"], level = b["level"]) for
    b in values(data["bus"]) if b["base_kv"] > 150
]
upper = @vlplot(
    data = (values = ehv,),
    mark = {:point, shape = "circle", size = 110, filled = true},
    x = {"x:q", axis = nothing, scale = {zero = false}},
    y = {"y:q", axis = nothing, scale = {zero = false}},
    color = {"level:n", scale = LEVEL_SCALE, legend = nothing},
)

# The two study buses, ringed and named. Colour is already spoken for by voltage
# level, so the marking uses a different channel. Every annotation is a flat
# single-layer spec: VegaLite's `+` drops layers when handed a nested layered one.
b_other, b_obs = data["bus"]["$(OTHER)"], data["bus"]["$(OBSERVED)"]

# One colour per marked bus, so the two are told apart without reading the labels.
const MARK_OTHER = "#D55E00"   # orange
const MARK_OBS = "#D62728"     # red, for the bus whose voltage the study watches

ring(bus, colour) = @vlplot(
    data = (values = [(x = bus["xcoord_1"], y = bus["ycoord_1"])],),
    mark = {
        :point,
        shape = "circle",
        size = 520,
        filled = false,
        strokeWidth = 2.2,
        stroke = colour,
    },
    x = {"x:q", axis = nothing, scale = {zero = false}},
    y = {"y:q", axis = nothing, scale = {zero = false}},
)

caption(bus, text, dx, dy, align, colour) = @vlplot(
    data = (values = [(x = bus["xcoord_1"], y = bus["ycoord_1"], t = text)],),
    mark = {:text, dx = dx, dy = dy, align = align, fontSize = 10, color = colour},
    x = {"x:q", axis = nothing, scale = {zero = false}},
    y = {"y:q", axis = nothing, scale = {zero = false}},
    text = {"t:n"},
)

# `save` renders at the converter's default 72 ppi, which on a 470 by 380 spec gives a
# 470 by 380 png: sharp on screen at 1:1 and blurry anywhere else. Rendering through
# `show` with a `:ppi` context scales the raster without touching the drawn size, so
# the layout, fonts and line weights stay exactly as specified.
const PPI = 288

function save_png(path, spec; ppi = PPI)
    open(path, "w") do io
        show(IOContext(io, :ppi => ppi), MIME"image/png"(), spec)
    end
    return path
end

figure =
    p +
    upper +
    ring(b_other, MARK_OTHER) +
    ring(b_obs, MARK_OBS) +
    caption(b_other, "farm $(OTHER)", 0, 26, "center", MARK_OTHER) +
    caption(b_obs, "farm $(OBSERVED)", 22, 4, "left", MARK_OBS)

out = save_png(joinpath(FIGDIR, "network.png"), figure)
println("Wrote ", relpath(out, pwd()), " at $(PPI) ppi")
