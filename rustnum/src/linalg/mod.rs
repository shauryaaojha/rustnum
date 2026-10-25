//! Linear algebra, like `numpy.linalg`.
//!
//! `Array` methods `matmul`, `dot` and `trace` live in this module's implementation; the free
//! functions here (`inv`, `solve`, `svd`, ...) accept `f32` and `f64` arrays. Numerical kernels
//! work on plain slices in `kernels` and are covered by NumPy-verified tests.
//!
//! ```
//! use rustnum as np;
//! let a = np::array![[4.0, 7.0], [2.0, 6.0]];
//! let ainv = np::linalg::inv(&a).unwrap();
//! assert!(a.matmul(&ainv).allclose(np::eye::<f64>(2), 1e-12, 1e-12));
//! ```

mod api;
pub(crate) mod kernels;

pub use api::*;
pub use kernels::NormOrd;

#[cfg(test)]
mod tests;
