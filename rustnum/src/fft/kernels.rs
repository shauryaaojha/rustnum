//! Slice-level FFT kernels. Pure Rust, no `Array` dependency.
//!
//! Complex data is stored as split `re` / `im` slices at the module boundary and as an
//! interleaved private [`Complex`] buffer internally. Twiddle factors are always computed in
//! `f64` and rounded once, so `f32` transforms do not accumulate trigonometric error.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// The few float operations the kernels need; implemented for `f32` and `f64` only.
pub(crate) trait Real:
    Copy
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    const ZERO: Self;
    fn from_f64(v: f64) -> Self;
    fn from_usize(n: usize) -> Self;
}

impl Real for f64 {
    const ZERO: Self = 0.0;
    fn from_f64(v: f64) -> Self {
        v
    }
    fn from_usize(n: usize) -> Self {
        n as f64
    }
}

impl Real for f32 {
    const ZERO: Self = 0.0;
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn from_usize(n: usize) -> Self {
        n as f32
    }
}

/// A complex number with real part `re` and imaginary part `im`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Complex<F> {
    pub re: F,
    pub im: F,
}

impl<F: Real> Complex<F> {
    pub(crate) fn new(re: F, im: F) -> Self {
        Complex { re, im }
    }

    fn zero() -> Self {
        Complex::new(F::ZERO, F::ZERO)
    }

    fn conj(self) -> Self {
        Complex::new(self.re, -self.im)
    }

    fn scale(self, s: F) -> Self {
        Complex::new(self.re * s, self.im * s)
    }

    /// `exp(-2 pi i * num / den)`, computed in `f64` from the reduced fraction.
    fn unit(num: usize, den: usize) -> Self {
        // Exact values on the axes, so small integer inputs transform without rounding noise.
        let r = num % den;
        if r == 0 {
            return Complex::new(F::from_f64(1.0), F::ZERO);
        } else if 2 * r == den {
            return Complex::new(F::from_f64(-1.0), F::ZERO);
        } else if 4 * r == den {
            return Complex::new(F::ZERO, F::from_f64(-1.0));
        } else if 4 * r == 3 * den {
            return Complex::new(F::ZERO, F::from_f64(1.0));
        }
        let angle = -2.0 * std::f64::consts::PI * (r as f64) / (den as f64);
        let (s, c) = angle.sin_cos();
        Complex::new(F::from_f64(c), F::from_f64(s))
    }
}

impl<F: Real> Add for Complex<F> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Complex::new(self.re + o.re, self.im + o.im)
    }
}

impl<F: Real> Sub for Complex<F> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Complex::new(self.re - o.re, self.im - o.im)
    }
}

impl<F: Real> Mul for Complex<F> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Complex::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

/// Non-power-of-two lengths up to this size use the direct O(n^2) DFT, which is the most
/// accurate option and cheaper than Bluestein for tiny n.
const DIRECT_MAX: usize = 16;

// ---------------------------------------------------------------------------------------
// Core in-place transforms (forward, unnormalised)
// ---------------------------------------------------------------------------------------

/// In-place forward DFT `X[k] = sum_j x[j] exp(-2 pi i j k / n)`, any length.
fn forward<F: Real>(buf: &mut [Complex<F>]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        radix2(buf);
    } else if n <= DIRECT_MAX {
        direct(buf);
    } else {
        bluestein(buf);
    }
}

/// In-place unnormalised inverse DFT via `conj(forward(conj(x)))`.
fn inverse_unscaled<F: Real>(buf: &mut [Complex<F>]) {
    for v in buf.iter_mut() {
        *v = v.conj();
    }
    forward(buf);
    for v in buf.iter_mut() {
        *v = v.conj();
    }
}

/// Iterative radix-2 decimation-in-time. `buf.len()` must be a power of two.
fn radix2<F: Real>(buf: &mut [Complex<F>]) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two());
    if n <= 1 {
        return;
    }
    // Bit-reversal permutation.
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            buf.swap(i, j);
        }
    }
    // Twiddles w[k] = exp(-2 pi i k / n), k < n / 2.
    let tw: Vec<Complex<F>> = (0..n / 2).map(|k| Complex::unit(k, n)).collect();
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = n / len;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let w = tw[k * step];
                let a = buf[start + k];
                let b = buf[start + k + half] * w;
                buf[start + k] = a + b;
                buf[start + k + half] = a - b;
            }
        }
        len <<= 1;
    }
}

/// Direct O(n^2) DFT with an exact twiddle table (`(j*k) mod n` indexing).
fn direct<F: Real>(buf: &mut [Complex<F>]) {
    let n = buf.len();
    let tw: Vec<Complex<F>> = (0..n).map(|k| Complex::unit(k, n)).collect();
    let src = buf.to_vec();
    for (k, out) in buf.iter_mut().enumerate() {
        let mut acc = Complex::zero();
        for (j, &x) in src.iter().enumerate() {
            acc = acc + x * tw[(j * k) % n];
        }
        *out = acc;
    }
}

/// Bluestein's chirp-z algorithm: a length-n DFT as a circular convolution of
/// power-of-two length `m >= 2n - 1`.
fn bluestein<F: Real>(buf: &mut [Complex<F>]) {
    let n = buf.len();
    let m = (2 * n - 1).next_power_of_two();
    // chirp[k] = exp(-i pi k^2 / n) = exp(-2 pi i (k^2 mod 2n) / (2n)).
    let chirp: Vec<Complex<F>> = (0..n)
        .map(|k| Complex::unit((k * k) % (2 * n), 2 * n))
        .collect();

    let mut a = vec![Complex::zero(); m];
    for k in 0..n {
        a[k] = buf[k] * chirp[k];
    }
    let mut b = vec![Complex::zero(); m];
    b[0] = chirp[0].conj();
    for k in 1..n {
        let c = chirp[k].conj();
        b[k] = c;
        b[m - k] = c;
    }

    radix2(&mut a);
    radix2(&mut b);
    for (x, y) in a.iter_mut().zip(&b) {
        *x = *x * *y;
    }
    // Inverse radix-2 of length m (conjugation trick) with 1/m scaling.
    for v in a.iter_mut() {
        *v = v.conj();
    }
    radix2(&mut a);
    let inv_m = F::from_f64(1.0 / m as f64);
    for k in 0..n {
        buf[k] = (a[k].conj().scale(inv_m)) * chirp[k];
    }
}

// ---------------------------------------------------------------------------------------
// Split-slice helpers
// ---------------------------------------------------------------------------------------

fn pack<F: Real>(re: &[F], im: &[F]) -> Vec<Complex<F>> {
    assert_eq!(
        re.len(),
        im.len(),
        "real and imaginary parts must have the same length"
    );
    re.iter()
        .zip(im)
        .map(|(&r, &i)| Complex::new(r, i))
        .collect()
}

fn unpack<F: Real>(buf: &[Complex<F>]) -> (Vec<F>, Vec<F>) {
    (
        buf.iter().map(|c| c.re).collect(),
        buf.iter().map(|c| c.im).collect(),
    )
}

fn scale_all<F: Real>(buf: &mut [Complex<F>]) {
    if buf.is_empty() {
        return;
    }
    let s = F::from_f64(1.0 / buf.len() as f64);
    for v in buf.iter_mut() {
        *v = v.scale(s);
    }
}

// ---------------------------------------------------------------------------------------
// Public kernels (crate-internal)
// ---------------------------------------------------------------------------------------

/// Forward DFT of a complex signal given as `(re, im)`, like `np.fft.fft`.
pub(crate) fn fft<F: Real>(re: &[F], im: &[F]) -> (Vec<F>, Vec<F>) {
    let mut buf = pack(re, im);
    forward(&mut buf);
    unpack(&buf)
}

/// Inverse DFT with `1/n` scaling, like `np.fft.ifft`.
pub(crate) fn ifft<F: Real>(re: &[F], im: &[F]) -> (Vec<F>, Vec<F>) {
    let mut buf = pack(re, im);
    inverse_unscaled(&mut buf);
    scale_all(&mut buf);
    unpack(&buf)
}

/// Forward DFT of a real signal; returns the `n / 2 + 1` non-redundant bins, like `np.fft.rfft`.
pub(crate) fn rfft<F: Real>(x: &[F]) -> (Vec<F>, Vec<F>) {
    let n = x.len();
    let mut buf: Vec<Complex<F>> = x.iter().map(|&v| Complex::new(v, F::ZERO)).collect();
    forward(&mut buf);
    buf.truncate(n / 2 + 1);
    unpack(&buf)
}

/// Inverse of [`rfft`]: `m` Hermitian half-spectrum bins to a real signal of length `n`,
/// like `np.fft.irfft(a, n)`. The half-spectrum is cropped or zero-padded to `n / 2 + 1`
/// bins; the imaginary parts of the DC (and Nyquist, for even `n`) bins are ignored.
pub(crate) fn irfft<F: Real>(re: &[F], im: &[F], n: usize) -> Vec<F> {
    assert_eq!(
        re.len(),
        im.len(),
        "real and imaginary parts must have the same length"
    );
    assert!(n >= 1, "Invalid number of FFT data points ({n}) specified.");
    let half = n / 2 + 1;
    let mut full = vec![Complex::<F>::zero(); n];
    for k in 0..half.min(re.len()) {
        full[k] = Complex::new(re[k], im[k]);
    }
    full[0].im = F::ZERO;
    if n.is_multiple_of(2) {
        full[n / 2].im = F::ZERO;
    }
    for k in 1..n.div_ceil(2) {
        full[n - k] = full[k].conj();
    }
    inverse_unscaled(&mut full);
    let s = F::from_f64(1.0 / n as f64);
    full.iter().map(|c| c.re * s).collect()
}

/// 2-D forward DFT of a row-major `rows x cols` complex array, like `np.fft.fft2`.
pub(crate) fn fft2<F: Real>(re: &[F], im: &[F], rows: usize, cols: usize) -> (Vec<F>, Vec<F>) {
    let mut buf = pack(re, im);
    assert_eq!(buf.len(), rows * cols, "shape does not match data length");
    transform_2d(&mut buf, rows, cols, false);
    unpack(&buf)
}

/// 2-D inverse DFT (scaled by `1 / (rows * cols)`), like `np.fft.ifft2`.
pub(crate) fn ifft2<F: Real>(re: &[F], im: &[F], rows: usize, cols: usize) -> (Vec<F>, Vec<F>) {
    let mut buf = pack(re, im);
    assert_eq!(buf.len(), rows * cols, "shape does not match data length");
    transform_2d(&mut buf, rows, cols, true);
    unpack(&buf)
}

/// Row pass then column pass. The inverse scales each pass by its own length.
fn transform_2d<F: Real>(buf: &mut [Complex<F>], rows: usize, cols: usize, inv: bool) {
    let run = |line: &mut [Complex<F>]| {
        if inv {
            inverse_unscaled(line);
            scale_all(line);
        } else {
            forward(line);
        }
    };
    if cols > 0 {
        for row in buf.chunks_mut(cols) {
            run(row);
        }
    }
    let mut col = vec![Complex::zero(); rows];
    for c in 0..cols {
        for r in 0..rows {
            col[r] = buf[r * cols + c];
        }
        run(&mut col);
        for r in 0..rows {
            buf[r * cols + c] = col[r];
        }
    }
}

/// Sample frequencies of an `n`-point DFT with sample spacing `d`, like `np.fft.fftfreq(n, d)`.
pub(crate) fn fftfreq<F: Real>(n: usize, d: F) -> Vec<F> {
    assert!(n > 0, "n should be an integer greater than 0");
    let val = F::from_usize(1) / (F::from_usize(n) * d);
    let pos = (n - 1) / 2 + 1;
    let mut out = Vec::with_capacity(n);
    for i in 0..pos {
        out.push(F::from_usize(i) * val);
    }
    for i in (1..=n / 2).rev() {
        out.push(-F::from_usize(i) * val);
    }
    out
}

/// Sample frequencies for `rfft` of an `n`-point signal, like `np.fft.rfftfreq(n, d)`.
pub(crate) fn rfftfreq<F: Real>(n: usize, d: F) -> Vec<F> {
    assert!(n > 0, "n should be an integer greater than 0");
    let val = F::from_usize(1) / (F::from_usize(n) * d);
    (0..=n / 2).map(|i| F::from_usize(i) * val).collect()
}

/// Move the zero-frequency bin to the centre on every axis of a row-major array of
/// `shape`, like `np.fft.fftshift(x)` (all axes). `inverse = true` gives `ifftshift`.
pub(crate) fn shift_nd<T: Copy>(data: &[T], shape: &[usize], inverse: bool) -> Vec<T> {
    let size: usize = shape.iter().product();
    assert_eq!(data.len(), size, "shape does not match data length");
    if size == 0 {
        return Vec::new();
    }
    // Destination offset along an axis of length n: fftshift rolls by n/2, ifftshift by -(n/2).
    let roll = |n: usize| if inverse { n - n / 2 } else { n / 2 };
    let nd = shape.len();
    let mut strides = vec![1usize; nd];
    for ax in (0..nd.saturating_sub(1)).rev() {
        strides[ax] = strides[ax + 1] * shape[ax + 1];
    }
    let mut out = data.to_vec();
    let mut idx = vec![0usize; nd];
    for &v in data {
        let dst: usize = (0..nd)
            .map(|ax| ((idx[ax] + roll(shape[ax])) % shape[ax]) * strides[ax])
            .sum();
        out[dst] = v;
        for ax in (0..nd).rev() {
            idx[ax] += 1;
            if idx[ax] < shape[ax] {
                break;
            }
            idx[ax] = 0;
        }
    }
    out
}

/// 1-D `np.fft.fftshift`.
pub(crate) fn fftshift<T: Copy>(x: &[T]) -> Vec<T> {
    shift_nd(x, &[x.len()], false)
}

/// 1-D `np.fft.ifftshift`.
pub(crate) fn ifftshift<T: Copy>(x: &[T]) -> Vec<T> {
    shift_nd(x, &[x.len()], true)
}

/// Crop or zero-pad a complex signal to length `n`, as NumPy does for the `n` argument.
pub(crate) fn resize<F: Real>(re: &[F], im: &[F], n: usize) -> (Vec<F>, Vec<F>) {
    let mut r = re.to_vec();
    let mut i = im.to_vec();
    r.resize(n, F::ZERO);
    i.resize(n, F::ZERO);
    (r, i)
}
