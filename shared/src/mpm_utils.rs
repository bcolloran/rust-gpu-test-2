use spirv_std::glam::Vec2;

#[inline(always)]
/// compute the weights for quadratic B-spline at position x, where x is scaled relative to a unit-sized grid cell.
/// "The Material Point Method for Simulating Continuum Materials" Eqn. 123
///
/// This simple reference implementation can be used to verify correctness of more optimized versions.
///
/// FIXME: Since we'll only be computing weights for grid cells at known offsets (-1, 0, 1), we can optimize this function further by only computing weights for these offsets instead of the full quadratic B-spline. When computing for these known offsets, we can avoid branches by using arithmetic expressions.
pub fn quadratic_weight(x: f32) -> f32 {
    let x_abs = x.abs();
    if x_abs < 0.5 {
        0.75 - x * x
    } else if x_abs < 1.5 {
        0.5 * (1.5 - x) * (1.5 - x)
    } else {
        0.0
    }
}

#[inline(always)]
/// compute the weight for quadratic B-spline at position fx (2D), where fx is scaled relative to a unit-sized grid cell.
///
/// FIXME: since this is a separable product of 1D weights, three 1D weights can be computed and reused instead of computing 2D weights directly.
pub fn quadratic_weight_2d(fx: Vec2) -> f32 {
    quadratic_weight(fx.x) * quadratic_weight(fx.y)
}
