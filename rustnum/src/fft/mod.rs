//! Discrete Fourier transforms, like `np.fft`.
//!
//! v0.1 has no complex dtype, so a complex signal is represented as a pair of real
//! arrays `(re, im)`. The numerical kernels live in the private `kernels` module and work on plain
//! slices; the public `Array`-level wrappers live in `api.rs`.
//!
//! All transforms use NumPy's default `"backward"` normalisation: the forward transform is
//! unscaled and the inverse divides by `n`. Every length is O(n log n): powers of two use an
//! iterative radix-2 Cooley-Tukey, other lengths use Bluestein's chirp-z algorithm.

mod api;
pub(crate) mod kernels;

pub use api::{fft, fft2, fftfreq, fftshift, ifft, ifft2, ifftshift, irfft, rfft, rfftfreq};

#[cfg(test)]
mod tests;
