# Display. The types that carry a network or a sample set hold megabytes of data,
# and the default struct display prints all of it on one line. Every such type gets
# a two-argument `show` that stays on one line for interpolation and containers, and
# a `text/plain` method that prints the tree below, in the style JuMP uses for its
# models.
#
#     Header
#     ├ key: value
#     ├ branch
#     │ ├ key: value
#     │ └ key: value
#     └ key: value
#
# An entry is a string, or a `String => Vector` whose vector holds its children.

const TREE_LIMIT = 5   # entries listed before a list is summarized instead

function show_tree(io::IO, header, entries::AbstractVector)
    print(io, header)
    show_entries(io, "", entries)
    return nothing
end

function show_entries(io::IO, prefix::AbstractString, entries::AbstractVector)
    for (k, entry) in enumerate(entries)
        # `last` would shadow Base.last, which the Pair branch below needs
        is_last = k == length(entries)
        print(io, "\n", prefix, is_last ? "└ " : "├ ")
        if entry isa Pair
            print(io, first(entry))
            show_entries(io, prefix * (is_last ? "  " : "│ "), last(entry))
        else
            print(io, entry)
        end
    end
    return nothing
end

# The first TREE_LIMIT items of a long list, with the remainder as a count. Listing
# every germ variable of a thousand-variable model is the problem, not the fix.
function listing(items, render = string)
    n = length(items)
    n <= TREE_LIMIT && return [render(x) for x in items]
    shown = [render(items[k]) for k = 1:TREE_LIMIT]
    push!(shown, "$(n - TREE_LIMIT) more")
    return shown
end

# Quantities of interest print unqualified, so a display reads the same inside a
# module that imports this package and at the prompt.
Base.show(io::IO, q::VoltageMagnitude) = print(io, "VoltageMagnitude(", q.bus, ")")
Base.show(io::IO, q::VoltageAngle) = print(io, "VoltageAngle(", q.bus, ")")
Base.show(io::IO, q::BranchActivePower) =
    print(io, "BranchActivePower(", q.from, ", ", q.to, ")")
Base.show(io::IO, q::ViolationEvent) =
    print(io, "ViolationEvent(", q.qoi, ", ", q.lo, ", ", q.hi, ")")

# Methods. Small enough to print on one line, but the positional form gives no clue
# which symbol is the warm-start mode and which is the failure policy.
method_entries(m::MonteCarlo) = [
    "n: $(m.n)",
    "failure_policy: $(repr(m.failure_policy))",
    "warmstart: $(repr(m.warmstart))",
    "keep_inputs: $(m.keep_inputs)",
]
method_entries(m::Union{LatinHypercube,SobolSampling}) =
    ["n: $(m.n)", "warmstart: $(repr(m.warmstart))", "keep_inputs: $(m.keep_inputs)"]
method_entries(m::QMCSampling) = [
    "sampler: $(nameof(typeof(m.sampler)))",
    "n: $(m.n)",
    "warmstart: $(repr(m.warmstart))",
    "keep_inputs: $(m.keep_inputs)",
]

for M in (:MonteCarlo, :LatinHypercube, :SobolSampling, :QMCSampling)
    @eval Base.show(io::IO, m::$M) = print(io, $(string(M)), "(n = ", m.n, ")")
    @eval Base.show(io::IO, ::MIME"text/plain", m::$M) =
        show_tree(io, $(string(M)), method_entries(m))
end

# Uncertainty model
Base.show(io::IO, c::GaussianCopula) = print(io, "GaussianCopula(d = ", size(c.L, 1), ")")
Base.show(io::IO, ::IndependentCopula) = print(io, "IndependentCopula()")

Base.show(io::IO, v::GermVariable) = print(io, "GermVariable(", repr(v.id), ")")
Base.show(io::IO, ::MIME"text/plain", v::GermVariable) =
    print(io, "GermVariable ", repr(v.id), ": ", v.dist)

Base.show(io::IO, r::ComponentRef) = print(io, r.kind, "[", repr(r.id), "].", r.field)

function Base.show(io::IO, a::Assignment)
    print(io, repr(a.variable), " → ", a.target)
    a.transform isa IdentityTransform || print(io, " via ", nameof(typeof(a.transform)))
    return nothing
end

Base.show(io::IO, m::UncertaintyModel) = print(
    io,
    "UncertaintyModel(",
    germ_dim(m),
    " germ variables, ",
    length(m.assignments),
    " assignments)",
)

Base.show(io::IO, ::MIME"text/plain", m::UncertaintyModel) =
    show_tree(io, "UncertaintyModel", model_entries(m))

model_entries(m::UncertaintyModel) = [
    "germ variables: $(germ_dim(m))" => listing(m.variables, v -> "$(v.id): $(v.dist)"),
    "assignments: $(length(m.assignments))" => listing(m.assignments),
    "dependence: $(m.dependence)",
]

# A solver is either a symbol or an algorithm object whose printed form is its full
# parametrized type, tens of type parameters wide. Ecosystem algorithms carry their
# own short name in a field, so prefer that.
solver_name(s::Symbol) = repr(s)
solver_name(s) =
    hasproperty(s, :name) ? string(getproperty(s, :name)) : string(nameof(typeof(s)))

# Problem
Base.show(io::IO, p::PPFProblem) = print(
    io,
    "PPFProblem(",
    sprint(show, p.backend),
    ", ",
    germ_dim(p.model),
    " germ variables, ",
    length(p.qois),
    " qois)",
)

Base.show(io::IO, ::MIME"text/plain", p::PPFProblem) = show_tree(
    io,
    "PPFProblem",
    [
        "backend: $(p.backend)",
        "model: $(sprint(show, p.model))" => model_entries(p.model),
        "quantities of interest: $(length(p.qois))" => listing(p.qois),
    ],
)

# Result
result_type(r::PPFResult) = string(nameof(PPFResult), "{", nameof(typeof(r.method)), "}")

Base.show(io::IO, r::PPFResult) =
    print(io, result_type(r), "(", n_converged(r), "/", r.n_samples, " converged)")

function Base.show(io::IO, ::MIME"text/plain", r::PPFResult)
    failures = if isempty(r.failures)
        "failures: none"
    else
        "failures: $(length(r.failures)), $(round(100 * failure_rate(r), digits = 1))% of the budget"
    end
    return show_tree(
        io,
        result_type(r),
        [
            "method: $(r.method)" => method_entries(r.method),
            "samples: $(n_converged(r)) converged of $(r.n_samples), in $(r.n_solves) solves",
            failures,
            "inputs kept: $(r.u === nothing ? "no" : "yes")",
            "quantities of interest: $(length(r.qois))" => listing(r.qois),
        ],
    )
end

# Reference backend
Base.show(io::IO, net::NetworkData) =
    print(io, "NetworkData(", net.n, " buses, ", length(net.branches), " branches)")

Base.show(io::IO, ::MIME"text/plain", net::NetworkData) = show_tree(
    io,
    "NetworkData",
    [
        "buses: $(net.n)" => [
            "slack: $(count(==(3), net.bustype))",
            "PV: $(count(==(2), net.bustype))",
            "PQ: $(count(==(1), net.bustype))",
        ],
        "branches: $(length(net.branches))",
        "total load: $(round(sum(net.pd), digits = 4)) + $(round(sum(net.qd), digits = 4))im pu",
    ],
)

Base.show(io::IO, b::ReferenceBackend) = print(io, "ReferenceBackend(", b.net.n, " buses)")

Base.show(io::IO, ::MIME"text/plain", b::ReferenceBackend) = show_tree(
    io,
    "ReferenceBackend",
    ["network: $(b.net)", "tol: $(b.tol)", "maxiter: $(b.maxiter)"],
)

Base.show(io::IO, s::RefState) =
    print(io, "RefState(", length(s.vm), " buses, ", length(s.slots), " injection slots)")

Base.show(io::IO, ::MIME"text/plain", s::RefState) = show_tree(
    io,
    "RefState",
    [
        "buses: $(length(s.vm))",
        "injection slots: $(length(s.slots))",
        "unknowns: $(length(s.ang_idx)) angles, $(length(s.pq_idx)) magnitudes",
    ],
)

Base.show(io::IO, info::SolveInfo) = print(
    io,
    "SolveInfo(",
    info.converged ? "converged" : "diverged",
    ", ",
    info.iterations,
    " iterations, residual ",
    info.residual,
    ")",
)
