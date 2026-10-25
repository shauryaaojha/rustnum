# rustnum API addendum: `np::fft` and `np::poly`

Same rules as `docs/API.md`: names follow NumPy; when this file is silent, do what NumPy does.
These two modules were out of scope in PLAN.md section 2 ("Out of scope for v0.1") and are added
here as public modules next to `linalg`, `random` and `io`. They need `Array<T>`, `IntoArray<T>`
and `Float` from API.md sections 1 to 3 and nothing else (no `linalg`, no `random`).

## 15. FFT (`fft/`) -- `np::fft::*`

### 15.1 Complex data is a `(re, im)` pair (v0.1 limitation)
rustnum v0.1 has no complex dtype. Everywhere NumPy takes or returns a complex array, the Rust
functions take or return **two real arrays of identical shape**, `(re, im)`:

```rust
use rustnum as np;
let x = np::array![1.0, 2.0, 3.0, 4.0];
let (re, im) = np::fft::fft(&x, &np::zeros_like(&x));   // np.fft.fft(x)
let (back_re, back_im) = np::fft::ifft(&re, &im);       // np.fft.ifft(X)
```

For real input use `rfft` (returns `n/2 + 1` bins) or pass `zeros_like` as `im`. When a complex dtype
arrives, the pair API stays as a thin wrapper; the pair-taking signatures will not be removed in 0.x.

### 15.2 Functions
`F: Float` (`f32` or `f64`). `T: Element` for the shifts. All transforms run along the **last axis**
(`fft2` along the last two); leading axes are batch axes. Internally everything is computed in
`f64` (twiddle factors included) and rounded once on output, also for `f32` arrays.

| Function | NumPy | Returns | Notes |
|---|---|---|---|
| `fft(&re, &im)` | `np.fft.fft(a)` | `(Array<F>, Array<F>)` | unscaled forward transform |
| `ifft(&re, &im)` | `np.fft.ifft(a)` | `(Array<F>, Array<F>)` | divides by `n` |
| `rfft(&a)` | `np.fft.rfft(a)` | `(Array<F>, Array<F>)` | real input; last axis becomes `n/2 + 1` |
| `irfft(&re, &im, n)` | `np.fft.irfft(a, n)` | `Array<F>` | `n` explicit (NumPy default is `2*(m-1)`); input cropped or zero-padded to `n/2 + 1` bins; imaginary parts of the DC and (even `n`) Nyquist bins are ignored like NumPy |
| `fft2(&re, &im)` | `np.fft.fft2(a)` | `(Array<F>, Array<F>)` | needs ndim >= 2; row pass then column pass |
| `ifft2(&re, &im)` | `np.fft.ifft2(a)` | `(Array<F>, Array<F>)` | divides by `rows * cols` |
| `fftfreq(n: usize, d: F)` | `np.fft.fftfreq(n, d)` | `Array<F>` | `[0, 1, .., (n-1)/2, -(n/2), .., -1] / (d*n)` |
| `rfftfreq(n: usize, d: F)` | `np.fft.rfftfreq(n, d)` | `Array<F>` | `[0, 1, .., n/2] / (d*n)` |
| `fftshift(&a)` | `np.fft.fftshift(a)` | `Array<T>` | all axes, any element type |
| `ifftshift(&a)` | `np.fft.ifftshift(a)` | `Array<T>` | inverse of `fftshift` |

### 15.3 Conventions and differences from NumPy
- Normalisation is NumPy's default `"backward"`: forward unscaled, inverse scaled by `1/n`.
  There is no `norm=` argument.
- No `n=` (pad/crop) or `axis=` arguments on `fft`/`ifft`/`rfft`. To pad or crop, build the input first
  (`concatenate` with zeros, or a slice). `irfft` is the exception: `n` is required (see table).
- Any length works in O(n log n): powers of two use iterative radix-2 Cooley-Tukey; other lengths use
  Bluestein's chirp-z algorithm (a direct O(n^2) DFT for non-powers of two up to n = 16, which is the
  most accurate and cheapest there). Length 1 is the identity; length 0 panics.
- Panics (NumPy wording where one exists): `Invalid number of FFT data points (0) specified.`,
  `axes exceeds dimensionality of input` (`fft2` on ndim < 2), and
  `real and imaginary parts must have the same shape, got [..] and [..]`.
- Accuracy (f64): `ifft(fft(x))` and the NumPy parity fixtures agree to about 1e-15 relative to the
  largest magnitude for n up to 1024, including primes.

### 15.4 Example
```rust
use rustnum as np;
let x = np::array![0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];   // 2 cycles in 8 samples
let (re, im) = np::fft::rfft(&x);
let freqs = np::fft::rfftfreq(8, 1.0);                          // [0, 0.125, 0.25, 0.375, 0.5]
// |X[k]| = hypot(re[k], im[k]) peaks at k = 2, i.e. 0.25 cycles per sample.
```

## 16. Polynomials (`poly/`) -- `np::poly::*`

NumPy's legacy `np.poly*` functions. A polynomial is a **1-d array of coefficients, highest power
first**: `array![1.0, -3.0, 2.0]` is `x^2 - 3x + 2`. Polynomial arguments take `impl IntoArray<F>`
(`F: Float`), so `&Array`, `Vec`, slices and arrays all work; non-1-d input panics
(`p must be a rank-1 array (got 2 dimensions)`). Computation is in `f64` internally.
There is no `poly1d` class and no `np.polynomial.Polynomial` in v0.1.

| Function | NumPy | Returns | Notes |
|---|---|---|---|
| `polyval(p, x)` | `np.polyval(p, x)` | `Array<F>` (shape of `x`) | Horner; `x` may be a scalar (result is 0-d, use `.item()`) |
| `polyfit(x, y, deg: usize)` | `np.polyfit(x, y, deg)` | `Array<F>` (`deg + 1`) | least squares; see 16.2 |
| `polyadd(a, b)` | `np.polyadd(a, b)` | `Array<F>` | constant terms aligned; leading zeros kept, as NumPy |
| `polysub(a, b)` | `np.polysub(a, b)` | `Array<F>` | same |
| `polymul(a, b)` | `np.polymul(a, b)` | `Array<F>` | convolution; empty input panics |
| `polydiv(u, v)` | `np.polydiv(u, v)` | `(Array<F>, Array<F>)` = `(q, r)` | NumPy algorithm incl. trimming `|r[0]| <= 1e-8`; leading `v[0] == 0` panics |
| `polyder(p, m: usize)` | `np.polyder(p, m)` | `Array<F>` | `m = 0` returns `p`; derivative of a constant is empty (as NumPy) |
| `polyint(p, m: usize, k: &[F])` | `np.polyint(p, m, k)` | `Array<F>` (`len + m`) | `k` empty = zeros, one value repeated, else `k.len() >= m` or panic `k must be a scalar or a rank-1 array of length 1 or >m.` |
| `roots(p)` | `np.roots(p)` | `(Array<F>, Array<F>)` = `(re, im)` | companion-matrix eigenvalues; see 16.3 |

### 16.1 Complex roots are a `(re, im)` pair
Same v0.1 limitation as 15.1. `roots` always returns two arrays of length `degree`; NumPy would return a
real array when every root is real, so check `im.iter().all(|v| v == 0.0)` for that case.

### 16.2 `polyfit`
Builds the Vandermonde matrix, scales each column to unit 2-norm (as NumPy does), and solves with
a column-pivoted Householder QR written in this module (no `linalg` dependency). Differences:
- Rank-deficient systems (fewer points than `deg + 1`, repeated abscissae): NumPy emits `RankWarning`
  and returns the *minimum-norm* least-squares solution. rustnum returns a *basic* solution (free
  coefficients set to 0) which fits the data equally well but has different coefficients.
- Not supported in v0.1: `w=` weights, `cov=`, `full=`, 2-d `y`.
- Panics: `expected non-empty vector for x`, `expected x and y to have same length`.

### 16.3 `roots`
- Leading zero coefficients are dropped; trailing zeros become zero roots appended at the end (NumPy).
- Eigenvalues come from the companion matrix, balanced like LAPACK's `dgebal`, then the Francis
  double-shift QR algorithm with LAPACK-style real shifts. If QR fails to converge in 30 iterations
  an Aberth-Ehrlich iteration takes over.
- Order: complex-conjugate pairs have the **positive imaginary part first**; otherwise roots follow the
  diagonal of the real Schur form, which matches NumPy/LAPACK order in practically all cases (395 of
  400 random polynomials in our check) but is **not guaranteed**. Do not rely on order; sort if it matters.
- Accuracy equals NumPy's: round-off-level for well-conditioned polynomials, about 1e-9 for Wilkinson's
  degree-10 polynomial (NumPy gets about the same).

### 16.4 Example
```rust
use rustnum as np;
let p = np::poly::polymul(&np::array![1.0, -1.0], &np::array![1.0, -2.0]);   // (x-1)(x-2)
let (re, im) = np::poly::roots(&p);          // re ~ [2, 1], im = [0, 0]
let y = np::poly::polyval(&p, &re);          // ~ [0, 0]
let c = np::poly::polyfit(&np::array![0.0, 1.0, 2.0, 3.0], &np::array![1.0, 3.0, 7.0, 13.0], 2);   // ~ [1, 1, 1]
```

## 17. Testing contract for these modules
- Kernels (`fft/kernels.rs`, `poly/kernels.rs`) are pure slice code with unit tests that compare against
  constants computed by NumPy 2.3 (generated once with Python) plus properties:
  `ifft(fft(x)) = x` for every n in 1..=64 and primes up to 1009, Parseval, FFT of a delta, linearity,
  `polyfit` recovers exact polynomials, `roots` of `(x-1)(x-2)(x-3)` is `[3, 2, 1]`.
- The `Array` wrappers (`api.rs`) are thin; each has a runnable doc example. Parity fixtures from
  `tools/gen_fixtures.py` should add: `fft`/`rfft` of random 1-d and 2-d inputs for n in
  {1..20, 64, 97, 100, 1000}, `polyfit` degrees 1 to 8, and `roots` of random polynomials (compare as
  sets).
