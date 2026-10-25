//! NumPy's remaining element-wise math, window functions and a complex number type.
//!
//! Step A (this module today): scalar kernels, windows and `Complex<F>`, tested against NumPy.
//! Step B (`api.rs.pending`): the public `Array` wrappers, enabled once the core `Array<T>` lands.
//!
//! Nothing here is re-exported yet, so the items are only used by the tests for now.
#![allow(dead_code)]

mod complex;
mod math_kernels;
mod window_kernels;

#[cfg(test)]
mod tests_complex;
#[cfg(test)]
mod tests_math;
#[cfg(test)]
mod tests_util;
#[cfg(test)]
mod tests_window;
