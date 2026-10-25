//! Polynomials with coefficients stored highest power first, like NumPy's legacy `np.poly*`
//! functions (`np.polyval`, `np.polyfit`, `np.roots`, ...).
//!
//! `[1.0, -3.0, 2.0]` is `x^2 - 3x + 2`. The numerical kernels live in the private `kernels`
//! module and work on plain `Vec<f64>` / `&[f64]`; the public `Array`-level wrappers live in
//! `api.rs`.
//!
//! Roots are complex in general. rustnum v0.1 has no complex dtype, so `roots` returns the
//! real and imaginary parts as two arrays `(re, im)`.

mod api;
pub(crate) mod kernels;

pub use api::{polyadd, polyder, polydiv, polyfit, polyint, polymul, polysub, polyval, roots};

#[cfg(test)]
mod tests;
