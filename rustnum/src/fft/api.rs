//! Public `np::fft` wrappers over `Array<T>`.
//!
//! Only slice-level kernels from `super::kernels` do the arithmetic; this layer converts
//! between `Array<F>` and `f64` buffers.
//!
//! v0.1 limitation: rustnum has no complex dtype. A complex array is a pair
//! `(re, im)` of same-shaped real arrays, both as inputs and as outputs. Arithmetic is done in
//! `f64` internally (also for `f32` arrays) and rounded once on the way out.

use super::kernels;
use crate::{Array, Float};

/// Flatten an array to `f64` in C order.
fn to_f64<F: Float>(a: &Array<F>) -> Vec<f64> {
    a.iter().map(|v| v.to_f64()).collect()
}

/// Rebuild an `Array<F>` of `shape` from `f64` data.
fn from_f64<F: Float>(data: Vec<f64>, shape: &[usize]) -> Array<F> {
    Array::from_vec(data.into_iter().map(F::from_f64).collect(), shape)
}

/// Panic unless `re` and `im` describe the same complex array.
fn check_pair<F: Float>(re: &Array<F>, im: &Array<F>) {
    assert!(
        re.shape() == im.shape(),
        "real and imaginary parts must have the same shape, got {:?} and {:?}",
        re.shape(),
        im.shape()
    );
}

/// Run `f` on every length-`n` line along the last axis, where `n_out(n)` is the output length.
fn along_last_axis<F: Float>(
    re: &Array<F>,
    im: &Array<F>,
    n_out: usize,
    f: impl Fn(&[f64], &[f64]) -> (Vec<f64>, Vec<f64>),
) -> (Array<F>, Array<F>) {
    check_pair(re, im);
    assert!(
        re.ndim() >= 1,
        "axis -1 is out of bounds for array of dimension 0"
    );
    let n = re.shape()[re.ndim() - 1];
    assert!(n >= 1, "Invalid number of FFT data points ({n}) specified.");
    let (xr, xi) = (to_f64(re), to_f64(im));
    let lines = xr.len() / n;
    let (mut or, mut oi) = (
        Vec::with_capacity(lines * n_out),
        Vec::with_capacity(lines * n_out),
    );
    for (lr, li) in xr.chunks_exact(n).zip(xi.chunks_exact(n)) {
        let (r, i) = f(lr, li);
        or.extend(r);
        oi.extend(i);
    }
    let mut shape = re.shape().to_vec();
    *shape.last_mut().unwrap() = n_out;
    (from_f64(or, &shape), from_f64(oi, &shape))
}

/// Discrete Fourier transform over the last axis, like `np.fft.fft(a)`.
///
/// The input is the complex array `re + i*im`; for a real signal pass `&np::zeros_like(&x)` as
/// `im` (or use [`rfft`]). Returns `(re, im)` of the same shape. Forward transforms are
/// unscaled (NumPy's default `"backward"` normalisation). Any length works in O(n log n).
///
/// **v0.1 limitation:** there is no complex dtype, so complex data is a `(re, im)` pair.
///
/// # Panics
/// If `re` and `im` differ in shape, the array is 0-d, or the last axis is empty.
///
/// # Examples
/// ```
/// use rustnum as np;
///
/// let x = np::array![1.0, 2.0, 3.0, 4.0];
/// let (re, im) = np::fft::fft(&x, &np::zeros_like(&x));
/// assert_eq!(re.to_vec(), vec![10.0, -2.0, -2.0, -2.0]);
/// assert_eq!(im.to_vec(), vec![0.0, 2.0, 0.0, -2.0]);
/// ```
pub fn fft<F: Float>(re: &Array<F>, im: &Array<F>) -> (Array<F>, Array<F>) {
    let n = re.shape().last().copied().unwrap_or(0);
    along_last_axis(re, im, n, kernels::fft::<f64>)
}

/// Inverse discrete Fourier transform over the last axis, like `np.fft.ifft(a)`.
///
/// Takes and returns a `(re, im)` pair; the result is divided by `n`, so
/// `ifft(fft(x)) == x` up to rounding.
///
/// # Panics
/// Same as [`fft`].
///
/// # Examples
/// ```
/// use rustnum as np;
///
/// let re = np::array![10.0, -2.0, -2.0, -2.0];
/// let im = np::array![0.0, 2.0, 0.0, -2.0];
/// let (xr, xi) = np::fft::ifft(&re, &im);
/// assert_eq!(xr.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(xi.to_vec(), vec![0.0; 4]);
/// ```
pub fn ifft<F: Float>(re: &Array<F>, im: &Array<F>) -> (Array<F>, Array<F>) {
    let n = re.shape().last().copied().unwrap_or(0);
    along_last_axis(re, im, n, kernels::ifft::<f64>)
}

/// FFT of a real signal over the last axis, like `np.fft.rfft(a)`.
///
/// Returns the `n / 2 + 1` non-redundant bins as `(re, im)`; the last axis of the result has
/// length `n / 2 + 1`.
///
/// **v0.1 limitation:** the complex result is returned as a `(re, im)` pair.
///
/// # Panics
/// If the array is 0-d or the last axis is empty.
///
/// # Examples
/// ```
/// use rustnum as np;
///
/// let (re, im) = np::fft::rfft(&np::array![1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(re.to_vec(), vec![10.0, -2.0, -2.0]);
/// assert_eq!(im.to_vec(), vec![0.0, 2.0, 0.0]);
/// ```
pub fn rfft<F: Float>(a: &Array<F>) -> (Array<F>, Array<F>) {
    let zeros = Array::from_vec(vec![F::from_f64(0.0); a.size()], a.shape());
    let n = a.shape().last().copied().unwrap_or(0);
    along_last_axis(a, &zeros, n / 2 + 1, |lr, _| kernels::rfft::<f64>(lr))
}

/// Inverse of [`rfft`], like `np.fft.irfft(a, n)`.
///
/// Takes the half-spectrum `(re, im)` and returns a real array whose last axis has length `n`.
/// NumPy's default is `n = 2 * (m - 1)` for `m` input bins; here `n` is explicit (there are no
/// keyword defaults), and it is required to recover odd lengths. The input is cropped or
/// zero-padded to `n / 2 + 1` bins, and the imaginary parts of the DC and (even `n`) Nyquist bins
/// are ignored, exactly as in NumPy.
///
/// **v0.1 limitation:** the complex input is a `(re, im)` pair.
///
/// # Panics
/// If `re` and `im` differ in shape, the array is 0-d, its last axis is empty, or `n == 0`.
///
/// # Examples
/// ```
/// use rustnum as np;
///
/// let re = np::array![10.0, -2.0, -2.0];
/// let im = np::array![0.0, 2.0, 0.0];
/// let x = np::fft::irfft(&re, &im, 4);
/// assert_eq!(x.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
/// ```
pub fn irfft<F: Float>(re: &Array<F>, im: &Array<F>, n: usize) -> Array<F> {
    assert!(n >= 1, "Invalid number of FFT data points ({n}) specified.");
    // The helper wants a pair of outputs; the imaginary half of a real result is just dropped.
    let (out, _) = along_last_axis(re, im, n, |lr, li| {
        (kernels::irfft::<f64>(lr, li, n), vec![0.0; n])
    });
    out
}

/// Two-dimensional DFT over the last two axes, like `np.fft.fft2(a)`.
///
/// Takes and returns a `(re, im)` pair. Leading axes (if any) are batch axes.
///
/// **v0.1 limitation:** complex data is a `(re, im)` pair.
///
/// # Panics
/// If `re` and `im` differ in shape or the array has fewer than 2 dimensions.
///
/// # Examples
/// ```
/// use rustnum as np;
///
/// let x = np::array![[1.0, 2.0], [3.0, 4.0]];
/// let (re, im) = np::fft::fft2(&x, &np::zeros_like(&x));
/// assert_eq!(re.to_vec(), vec![10.0, -2.0, -4.0, 0.0]);
/// assert_eq!(im.to_vec(), vec![0.0; 4]);
/// ```
pub fn fft2<F: Float>(re: &Array<F>, im: &Array<F>) -> (Array<F>, Array<F>) {
    transform2(re, im, kernels::fft2::<f64>)
}

/// Inverse two-dimensional DFT over the last two axes, like `np.fft.ifft2(a)`.
///
/// Takes and returns a `(re, im)` pair; divides by `rows * cols`.
///
/// # Panics
/// Same as [`fft2`].
///
/// # Examples
/// ```
/// use rustnum as np;
///
/// let re = np::array![[10.0, -2.0], [-4.0, 0.0]];
/// let (xr, xi) = np::fft::ifft2(&re, &np::zeros_like(&re));
/// assert_eq!(xr.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(xi.to_vec(), vec![0.0; 4]);
/// ```
pub fn ifft2<F: Float>(re: &Array<F>, im: &Array<F>) -> (Array<F>, Array<F>) {
    transform2(re, im, kernels::ifft2::<f64>)
}

fn transform2<F: Float>(
    re: &Array<F>,
    im: &Array<F>,
    f: impl Fn(&[f64], &[f64], usize, usize) -> (Vec<f64>, Vec<f64>),
) -> (Array<F>, Array<F>) {
    check_pair(re, im);
    assert!(re.ndim() >= 2, "axes exceeds dimensionality of input");
    let nd = re.ndim();
    let (rows, cols) = (re.shape()[nd - 2], re.shape()[nd - 1]);
    assert!(
        rows >= 1 && cols >= 1,
        "Invalid number of FFT data points (0) specified."
    );
    let (xr, xi) = (to_f64(re), to_f64(im));
    let block = rows * cols;
    let (mut or, mut oi) = (Vec::with_capacity(xr.len()), Vec::with_capacity(xr.len()));
    for (br, bi) in xr.chunks_exact(block).zip(xi.chunks_exact(block)) {
        let (r, i) = f(br, bi, rows, cols);
        or.extend(r);
        oi.extend(i);
    }
    (from_f64(or, re.shape()), from_f64(oi, re.shape()))
}

/// Sample frequencies of an `n`-point DFT with sample spacing `d`, like `np.fft.fftfreq(n, d)`.
///
/// Returns `[0, 1, ..., (n-1)/2, -(n/2), ..., -1] / (d * n)`.
///
/// # Panics
/// If `n == 0`.
///
/// # Examples
/// ```
/// use rustnum as np;
///
/// let f = np::fft::fftfreq(4, 0.5_f64);
/// assert_eq!(f.to_vec(), vec![0.0, 0.5, -1.0, -0.5]);
/// ```
pub fn fftfreq<F: Float>(n: usize, d: F) -> Array<F> {
    let v = kernels::fftfreq::<f64>(n, d.to_f64());
    from_f64(v, &[n])
}

/// Sample frequencies for [`rfft`] of an `n`-point signal, like `np.fft.rfftfreq(n, d)`.
///
/// Returns `[0, 1, ..., n/2] / (d * n)`, `n / 2 + 1` values.
///
/// # Panics
/// If `n == 0`.
///
/// # Examples
/// ```
/// use rustnum as np;
///
/// let f = np::fft::rfftfreq(4, 0.5_f64);
/// assert_eq!(f.to_vec(), vec![0.0, 0.5, 1.0]);
/// ```
pub fn rfftfreq<F: Float>(n: usize, d: F) -> Array<F> {
    let v = kernels::rfftfreq::<f64>(n, d.to_f64());
    let len = v.len();
    from_f64(v, &[len])
}

/// Shift the zero-frequency component to the centre, like `np.fft.fftshift(a)` (all axes).
///
/// Works for any element type. `ifftshift` is the inverse.
///
/// # Examples
/// ```
/// use rustnum as np;
///
/// let a = np::array![0.0, 1.0, 2.0, 3.0, 4.0];
/// assert_eq!(np::fft::fftshift(&a).to_vec(), vec![3.0, 4.0, 0.0, 1.0, 2.0]);
/// ```
pub fn fftshift<T: crate::Element>(a: &Array<T>) -> Array<T> {
    Array::from_vec(kernels::shift_nd(&a.to_vec(), a.shape(), false), a.shape())
}

/// Inverse of [`fftshift`], like `np.fft.ifftshift(a)` (all axes).
///
/// # Examples
/// ```
/// use rustnum as np;
///
/// let a = np::array![0.0, 1.0, 2.0, 3.0, 4.0];
/// assert_eq!(np::fft::ifftshift(&a).to_vec(), vec![2.0, 3.0, 4.0, 0.0, 1.0]);
/// ```
pub fn ifftshift<T: crate::Element>(a: &Array<T>) -> Array<T> {
    Array::from_vec(kernels::shift_nd(&a.to_vec(), a.shape(), true), a.shape())
}
