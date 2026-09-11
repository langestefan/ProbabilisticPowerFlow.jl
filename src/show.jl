const TREE_LIMIT = 5

function show_tree(io::IO, header, entries::AbstractVector)
    print(io, header)
    show_entries(io, "", entries)
    return nothing
end

function show_entries(io::IO, prefix::AbstractString, entries::AbstractVector)
    for (k, entry) in enumerate(entries)
        if k == length(entries)
            branch, indent = "└ ", "  "
        else
            branch, indent = "├ ", "│ "
        end
        print(io, "\n", prefix, branch)
        if entry isa Pair
            print(io, first(entry))
            show_entries(io, prefix * indent, last(entry))
        else
            print(io, entry)
        end
    end
    return nothing
end

function listing(items, render = string)
    n = length(items)
    if n <= TREE_LIMIT
        return [render(x) for x in items]
    end
    shown = [render(items[k]) for k in 1:TREE_LIMIT]
    push!(shown, "$(n - TREE_LIMIT) more")
    return shown
end

Base.show(io::IO, r::ComponentRef) =
    print(io, "ComponentRef(ComponentField.", Symbol(r.field), ", ", r.id, ")")

ref_label(r::ComponentRef) = "$(Symbol(r.field))[$(r.id)]"

function Base.show(io::IO, info::SolveInfo)
    if info.converged
        outcome = "converged"
    else
        outcome = "diverged"
    end
    print(io, "SolveInfo(", outcome, ", ", info.iterations, " iterations, residual ")
    print(io, info.residual, ")")
    return nothing
end

Base.show(io::IO, v::ViolationEvent) =
    print(io, "ViolationEvent(", v.qoi, ", ", v.lo, ", ", v.hi, ")")

Base.show(io::IO, v::GermVariable) =
    print(io, "GermVariable(", repr(v.id), ", ", v.dist, ")")

function Base.show(io::IO, a::Assignment)
    print(io, repr(a.variable), " → ", ref_label(a.target))
    if !(a.transform isa IdentityTransform)
        print(io, " via ", a.transform)
    end
    return nothing
end

dependence_label(c) = "$(nameof(typeof(c)))(d = $(length(c)))"

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
    "dependence: $(dependence_label(m.dependence))",
]

network_counts(data::AbstractDict) = [
    "buses: $(length(data["bus"]))",
    "branches: $(length(data["branch"]))",
    "generators: $(length(data["gen"]))",
    "loads: $(length(data["load"]))",
]

Base.show(io::IO, b::PowerModelsBackend) =
    print(io, "PowerModelsBackend(", length(b.data["bus"]), " buses)")

Base.show(io::IO, ::MIME"text/plain", b::PowerModelsBackend) = show_tree(
    io,
    "PowerModelsBackend",
    [
        "network: $(get(b.data, "name", "unnamed"))" => network_counts(b.data),
        "algorithm: $(nameof(typeof(b.alg)))",
    ],
)

Base.show(io::IO, p::PPFProblem) = print(
    io,
    "PPFProblem(",
    p.backend,
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
        "model: $(p.model)" => model_entries(p.model),
        "quantities of interest: $(length(p.qois))" => listing(p.qois),
    ],
)

Base.show(io::IO, m::MonteCarlo) = print(io, "MonteCarlo(n = ", m.n, ")")
Base.show(io::IO, m::LatinHypercube) = print(io, "LatinHypercube(n = ", m.n, ")")
Base.show(io::IO, m::QuasiMC) =
    print(io, "QuasiMC(", nameof(typeof(m.sampler)), ", n = ", m.n, ")")

result_type(r::PPFResult) = "PPFResult{$(nameof(typeof(r.method)))}"

Base.show(io::IO, r::PPFResult) =
    print(io, result_type(r), "(", n_converged(r), "/", r.n_samples, " converged)")

function Base.show(io::IO, ::MIME"text/plain", r::PPFResult)
    show_tree(
        io,
        result_type(r),
        [
            "method: $(r.method)",
            "samples: $(n_converged(r)) converged of $(r.n_samples), in $(r.n_solves) solves",
            failure_summary(r),
            "quantities of interest: $(length(r.qois))" => listing(r.qois),
        ],
    )
    return nothing
end

function failure_summary(r::PPFResult)
    if isempty(r.failures)
        return "failures: none"
    end
    percent = round(100 * failure_rate(r), digits = 1)
    return "failures: $(length(r.failures)), $(percent)% of samples"
end
