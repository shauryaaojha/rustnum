//! Discrete Fourier transforms, like `np.fft`.
//!
//! Like NumPy, every transform takes a real or complex array and returns a complex
//! `Array<Complex<F>>` (`complex128` for `f64`, `complex64` for `f32`). The numerical kernels
//! live in the private `kernels` module and work on plain slices; the public `Array`-level
//! wrappers live in `api.rs`.
//!
//! All transforms use NumPy's default `"backward"` normalisation: the forward transform is
//! unscaled and the inverse divides by `n`. Every length is O(n log n): powers of two use an
//! iterative radix-2 Cooley-Tukey, other lengths use Bluestein's chirp-z algorithm.

mod api;
pub(crate) mod kernels;

pub use api::{
    FftInput, fft, fft2, fftfreq, fftshift, ifft, ifft2, ifftshift, irfft, rfft, rfftfreq,
};

#[cfg(test)]
mod tests;
