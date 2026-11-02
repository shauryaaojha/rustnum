//! Public `np::fft` wrappers over `Array<T>`.
//!
//! Only slice-level kernels from `super::kernels` do the arithmetic; this layer converts
//! between `Array<Complex<F>>` and split `f64` buffers. Like NumPy, every transform accepts a
//! real or a complex array and returns a complex array. Arithmetic is done in `f64` internally
//! (also for `f32` arrays) and rounded once on the way out.

use super::kernels;
use crate::{Array, Complex, Float};

/// An array a Fourier transform can read: real (`Array<F>`) or complex (`Array<Complex<F>>`).
///
/// Real input is promoted to complex with zero imaginary part, like NumPy.
/// ```
/// use rustnum as np;
/// use np::fft::FftInput;
/// let z = np::array![1.0f64, 2.0].to_complex();
/// assert_eq!(z.to_vec(), [np::Complex::new(1.0, 0.0), np::Complex::new(2.0, 0.0)]);
/// ```
pub trait FftInput {
    /// Real scalar type of the transform (`f32` or `f64`).
    type Real: Float;
    /// The input as a complex array.
    /// ```
    /// use rustnum as np;
    /// use np::fft::FftInput;
    /// assert_eq!(np::array![1.0f32].to_complex().item(), np::Complex::new(1.0, 0.0));
    /// ```
    fn to_complex(&self) -> Array<Complex<Self::Real>>;
}
impl<F: Float> FftInput for Array<F> {
    type Real = F;
    fn to_complex(&self) -> Array<Complex<F>> {
        self.par_map(Complex::from_real)
    }
}
impl<F: Float> FftInput for Array<Complex<F>> {
    type Real = F;
    fn to_complex(&self) -> Array<Complex<F>> {
        self.clone()
    }
}

/// Split a complex array into `f64` real and imaginary buffers in C order.
fn split<F: Float>(a: &Array<Complex<F>>) -> (Vec<f64>, Vec<f64>) {
    let mut re = Vec::with_capacity(a.size());
    let mut im = Vec::with_capacity(a.size());
    for z in a.iter() {
        re.push(z.re.to_f64());
        im.push(z.im.to_f64());
    }
    (re, im)
}

/// Rebuild an `Array<Complex<F>>` of `shape` from split `f64` buffers.
fn join<F: Float>(re: Vec<f64>, im: Vec<f64>, shape: &[usize]) -> Array<Complex<F>> {
    Array::from_vec(
        re.into_iter()
            .zip(im)
            .map(|(r, i)| Complex::new(F::from_f64(r), F::from_f64(i)))
            .collect(),
        shape,
    )
}

/// Rebuild an `Array<F>` of `shape` from `f64` data.
fn from_f64<F: Float>(data: Vec<f64>, shape: &[usize]) -> Array<F> {
    Array::from_vec(data.into_iter().map(F::from_f64).collect(), shape)
}

/// Run `f` on every length-`n` line along the last axis, where `n_out` is the output length.
fn along_last_axis<F: Float>(
    a: &Array<Complex<F>>,
    n_out: usize,
    f: impl Fn(&[f64], &[f64]) -> (Vec<f64>, Vec<f64>),
) -> Array<Complex<F>> {
    assert!(
        a.ndim() >= 1,
        "axis -1 is out of bounds for array of dimension 0"
    );
    let n = a.shape()[a.ndim() - 1];
    assert!(n >= 1, "Invalid number of FFT data points ({n}) specified.");
    let (xr, xi) = split(a);
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
    let mut shape = a.shape().to_vec();
    *shape.last_mut().unwrap() = n_out;
    join(or, oi, &shape)
}

/// Discrete Fourier transform over the last axis, like `np.fft.fft(a)`.
///
/// Accepts a real or complex array and returns a complex array of the same shape. Forward
/// transforms are unscaled (NumPy's default `"backward"` normalisation). Any length works in
/// O(n log n).
///
/// # Panics
/// If the array is 0-d or the last axis is empty.
///
/// # Examples
/// ```
/// use rustnum as np;
///
/// let x = np::array![1.0, 2.0, 3.0, 4.0];
/// let y = np::fft::fft(&x);
/// assert_eq!(y.real().to_vec(), vec![10.0, -2.0, -2.0, -2.0]);
/// assert_eq!(y.imag().to_vec(), vec![0.0, 2.0, 0.0, -2.0]);
/// assert_eq!(format!("{y}"), "[10.+0.j -2.+2.j -2.+0.j -2.-2.j]");
/// ```
pub fn fft<A: FftInput>(a: &A) -> Array<Complex<A::Real>> {
    let a = a.to_complex();
    let n = a.shape().last().copied().unwrap_or(0);
    along_last_axis(&a, n, kernels::fft::<f64>)
}

/// Inverse discrete Fourier transform over the last axis, like `np.fft.ifft(a)`.
///
/// The result is divided by `n`, so `ifft(&fft(&x))` equals `x` up to rounding.
///
/// # Panics
/// Same as [`fft`].
///
/// # Examples
/// ```
/// use rustnum as np;
///
/// let x = np::array![1.0, 2.0, 3.0, 4.0];
/// let back = np::fft::ifft(&np::fft::fft(&x));
/// assert_eq!(back.real().to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(back.imag().to_vec(), vec![0.0; 4]);
/// ```
pub fn ifft<A: FftInput>(a: &A) -> Array<Complex<A::Real>> {
    let a = a.to_complex();
    let n = a.shape().last().copied().unwrap_or(0);
    along_last_axis(&a, n, kernels::ifft::<f64>)
}

/// FFT of a real signal over the last axis, like `np.fft.rfft(a)`.
///
/// Returns the `n / 2 + 1` non-redundant bins as a complex array; the last axis of the result
/// has length `n / 2 + 1`.
///
/// # Panics
/// If the array is 0-d or the last axis is empty.
///
/// # Examples
/// ```
/// use rustnum as np;
///
/// let y = np::fft::rfft(&np::array![1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(y.real().to_vec(), vec![10.0, -2.0, -2.0]);
/// assert_eq!(y.imag().to_vec(), vec![0.0, 2.0, 0.0]);
/// ```
pub fn rfft<F: Float>(a: &Array<F>) -> Array<Complex<F>> {
    let n = a.shape().last().copied().unwrap_or(0);
    along_last_axis(&a.to_complex(), n / 2 + 1, |lr, _| kernels::rfft::<f64>(lr))
}

/// Inverse of [`rfft`], like `np.fft.irfft(a, n)`.
///
/// Takes the half-spectrum and returns a real array whose last axis has length `n`. With
/// `n = None` the length is `2 * (m - 1)` for `m` input bins, NumPy's default; pass `Some(n)`
/// to recover odd lengths. The input is cropped or zero-padded to `n / 2 + 1` bins, and the
/// imaginary parts of the DC and (even `n`) Nyquist bins are ignored, exactly as in NumPy.
///
/// # Panics
/// If the array is 0-d, its last axis is empty, or `n == 0`.
///
/// # Examples
/// ```
/// use rustnum as np;
///
/// let y = np::fft::rfft(&np::array![1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(np::fft::irfft(&y, None).to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
/// let odd = np::fft::rfft(&np::array![1.0f64, 2.0, 3.0]);
/// let back = np::fft::irfft(&odd, Some(3)).to_vec();
/// assert!(back.iter().zip([1.0, 2.0, 3.0]).all(|(a, b)| (a - b).abs() < 1e-12));
/// ```
pub fn irfft<A: FftInput>(a: &A, n: Option<usize>) -> Array<A::Real> {
    let a = a.to_complex();
    assert!(
        a.ndim() >= 1,
        "axis -1 is out of bounds for array of dimension 0"
    );
    let m = a.shape()[a.ndim() - 1];
    let n = n.unwrap_or_else(|| 2 * m.saturating_sub(1));
    assert!(n >= 1, "Invalid number of FFT data points ({n}) specified.");
    // The helper wants a pair of outputs; the imaginary half of a real result is just dropped.
    let out = along_last_axis(&a, n, |lr, li| {
        (kernels::irfft::<f64>(lr, li, n), vec![0.0; n])
    });
    out.real()
}

/// Two-dimensional DFT over the last two axes, like `np.fft.fft2(a)`.
///
/// Accepts a real or complex array; leading axes (if any) are batch axes.
///
/// # Panics
/// If the array has fewer than 2 dimensions or an empty transform axis.
///
/// # Examples
/// ```
/// use rustnum as np;
///
/// let x = np::array![[1.0, 2.0], [3.0, 4.0]];
/// let y = np::fft::fft2(&x);
/// assert_eq!(y.real().to_vec(), vec![10.0, -2.0, -4.0, 0.0]);
/// assert_eq!(y.imag().to_vec(), vec![0.0; 4]);
/// ```
pub fn fft2<A: FftInput>(a: &A) -> Array<Complex<A::Real>> {
    transform2(&a.to_complex(), kernels::fft2::<f64>)
}

/// Inverse two-dimensional DFT over the last two axes, like `np.fft.ifft2(a)`.
///
/// Divides by `rows * cols`.
///
/// # Panics
/// Same as [`fft2`].
///
/// # Examples
/// ```
/// use rustnum as np;
///
/// let x = np::array![[1.0, 2.0], [3.0, 4.0]];
/// let back = np::fft::ifft2(&np::fft::fft2(&x));
/// assert_eq!(back.real().to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
/// ```
pub fn ifft2<A: FftInput>(a: &A) -> Array<Complex<A::Real>> {
    transform2(&a.to_complex(), kernels::ifft2::<f64>)
}

fn transform2<F: Float>(
    a: &Array<Complex<F>>,
    f: impl Fn(&[f64], &[f64], usize, usize) -> (Vec<f64>, Vec<f64>),
) -> Array<Complex<F>> {
    assert!(a.ndim() >= 2, "axes exceeds dimensionality of input");
    let nd = a.ndim();
    let (rows, cols) = (a.shape()[nd - 2], a.shape()[nd - 1]);
    assert!(
        rows >= 1 && cols >= 1,
        "Invalid number of FFT data points (0) specified."
    );
    let (xr, xi) = split(a);
    let block = rows * cols;
    let (mut or, mut oi) = (Vec::with_capacity(xr.len()), Vec::with_capacity(xr.len()));
    for (br, bi) in xr.chunks_exact(block).zip(xi.chunks_exact(block)) {
        let (r, i) = f(br, bi, rows, cols);
        or.extend(r);
        oi.extend(i);
    }
    join(or, oi, a.shape())
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
