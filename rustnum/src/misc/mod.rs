//! NumPy's remaining element-wise math, window functions and a complex number type.
//!
//! - `math_kernels`: scalar kernels written once over the core [`Float`](crate::Float) trait
//!   (plus its sealed supertrait `FloatHooks`), tested against NumPy 2.3.
//! - `window_kernels`: `hanning`, `hamming`, `blackman`, `bartlett`, `kaiser`.
//! - `complex`: `Complex<F>`.
//! - `api`: the public `Array` methods and free functions, re-exported from the crate root.

mod api;
mod complex;
mod math_kernels;
mod window_kernels;

pub use api::*;
pub use complex::Complex;
pub use math_kernels::FloatHooks;

#[cfg(test)]
mod tests_complex;
#[cfg(test)]
mod tests_math;
#[cfg(test)]
mod tests_util;
#[cfg(test)]
mod tests_window;
