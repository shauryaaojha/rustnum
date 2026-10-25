//! Polynomials with coefficients stored highest power first, like NumPy's legacy `np.poly*`
//! functions (`np.polyval`, `np.polyfit`, `np.roots`, ...).
//!
//! `[1.0, -3.0, 2.0]` is `x^2 - 3x + 2`. The numerical kernels live in the private `kernels`
//! module and work on plain `Vec<f64>` / `&[f64]`; the `Array`-level wrappers are written in
//! `api.rs.pending` and are enabled once the core `Array<T>` type lands.
//!
//! Roots are complex in general. rustnum v0.1 has no complex dtype, so `roots` returns the
//! real and imaginary parts as two arrays `(re, im)`.

// The kernels are exercised by the unit tests; the public wrappers are not compiled yet.
#[allow(dead_code)]
pub(crate) mod kernels;

#[cfg(test)]
mod tests;
