"""
    AbstractTransform

A deterministic map from a germ variable's value to a physical injection value.

An example is passing a sampled wind speed through a wind turbine power curve to get the
physical injection. Transforms are callable: `t(z) -> Float64`.
"""
abstract type AbstractTransform end

"""
    IdentityTransform()

The germ value maps directly to a physical injection.
"""
struct IdentityTransform <: AbstractTransform end

(::IdentityTransform)(z::Real) = float(z)

"""
    AffineTransform(a, b)

Injections of the form `z ↦ a * z + b`.
"""
struct AffineTransform <: AbstractTransform
    a::Float64
    b::Float64
end

(t::AffineTransform)(z::Real) = t.a * z + t.b
