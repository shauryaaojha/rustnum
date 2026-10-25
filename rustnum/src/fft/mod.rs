//! Discrete Fourier transforms, like `np.fft`.
//!
//! v0.1 has no complex dtype, so a complex signal is represented as a pair of real
//! arrays `(re, im)`. The numerical kernels live in the private `kernels` module and work on plain slices;
//! the `Array`-level wrappers are written in `api.rs.pending` and are enabled once the
//! core `Array<T>` type lands.
//!
//! All transforms use NumPy's default `"backward"` normalisation: the forward transform is
//! unscaled and the inverse divides by `n`. Every length is O(n log n): powers of two use an
//! iterative radix-2 Cooley-Tukey, other lengths use Bluestein's chirp-z algorithm.

// The kernels are exercised by the unit tests; the public wrappers are not compiled yet.
#[allow(dead_code)]
pub(crate) mod kernels;

#[cfg(test)]
mod tests;
