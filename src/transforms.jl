"""
    AbstractTransform

A deterministic map from a germ variable's value to a physical injection value.
Transforms are callable: `t(z) -> Float64`.

The serialized-spec vocabulary (`power_curve`, `pv_model`, `clip`) and the
`scale_base`/`add_base` assignment modes arrive together with the TOML loader; the
base-value modes additionally need a `base_value(backend, ref)` contract function to
read the pinned network's base values.
"""
abstract type AbstractTransform end

"""
    IdentityTransform()

The germ value is the injection.
"""
struct IdentityTransform <: AbstractTransform end

(::IdentityTransform)(z::Real) = float(z)

"""
    AffineTransform(a, b)

`z ↦ a * z + b`.
"""
struct AffineTransform <: AbstractTransform
    a::Float64
    b::Float64
end

(t::AffineTransform)(z::Real) = t.a * z + t.b
