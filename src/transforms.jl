"""
    AbstractTransform

A deterministic map from a germ variable's value to a physical injection value. We need
this to couple a random input to a physical injection in the units the backend expects,
for example wind speed to wind power, or a load forecast to a load injection.

Transforms are callable: `t(z) -> Float64`.

The TOML loader will add the remaining transform functions: `power_curve`,
`pv_model`, `clip`, and the `scale_base` and `add_base` assignment modes.

The base modes need to read base values from the network, which requires a new function
`base_value(backend, ref)`.
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
