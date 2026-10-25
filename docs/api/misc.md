# rustnum misc: element-wise math, windows, complex numbers

API section in the style of `docs/API.md`. Names follow NumPy. The code lives in
`rustnum/src/misc/`: scalar kernels and `Complex<F>` are implemented and tested against NumPy 2.3
today; the `Array` wrappers are written in `rustnum/src/misc/api.rs.pending` and switch on once the
core `Array<T>` exists.

## M0. Status and wiring

| Part | File | State |
|---|---|---|
| Scalar kernels (`F = f32 \| f64`) | `misc/math_kernels.rs` | compiled, 58 NumPy parity tests |
| Window kernels | `misc/window_kernels.rs` | compiled, 78 tests |
| `Complex<F>` | `misc/complex.rs` | compiled, 54 tests |
| `Array` wrappers | `misc/api.rs.pending` | written, not compiled (needs core `Array`) |

To enable the wrappers: rename `api.rs.pending` to `api.rs`, add `mod api; pub use api::{...};` to
`misc/mod.rs` and `pub use misc::{...};` to `lib.rs` for the names in M1 and M2 (one path per item,
rule 2 in `AGENTS.md`), then drop `#![allow(dead_code)]` from `misc/mod.rs`.

Wrapper bounds are `F: Element + KFloat`. `KFloat` is a private sealed trait implemented for `f32`
and `f64` only, so the bound means the same as `Float`; it avoids method-name ambiguity with the core
`Float` trait (`sqrt`, `from_f64`, `ZERO` exist on both). Generic user code over `F: Float` cannot
call these methods yet; use concrete `Array<f64>` / `Array<f32>` (see M5 for the clean-up).

## M1. Element-wise methods on `Array<F: Float>` (return a new array)

| Method | NumPy | Notes |
|---|---|---|
| `sinc()` | `np.sinc` | normalized; `sinc(0) = 1` (NumPy 2.3 substitutes `eps` for 0) |
| `i0()` | `np.i0` | Cephes Chebyshev coefficients exactly as NumPy; `i0(inf) = nan` |
| `deg2rad()`, `rad2deg()` | `np.deg2rad`, `np.rad2deg` | |
| `rint()` | `np.rint` | half to even, keeps `-0.0` |
| `fix()` | `np.fix` | toward zero, `fix(-0.5) = -0.0` |
| `fabs()`, `positive()`, `negative()` | `np.fabs`, `np.positive`, `np.negative` | `negative(0.0) = -0.0` |
| `signbit() -> Array<bool>` | `np.signbit` | true for `-0.0` and negative NaN |
| `spacing()` | `np.spacing` | `spacing(+-inf) = nan`, `spacing(-0.0) = +5e-324` |
| `modf() -> (frac, int)` | `np.modf` | both parts carry the sign of the input |
| `frexp() -> (Array<F>, Array<i32>)` | `np.frexp` | mantissa in `[0.5, 1)`; zero/inf/nan give exponent 0 |
| `nan_to_num()` | `np.nan_to_num(a)` | NaN to 0, +-inf to +-`F::MAX` |
| `nan_to_num_with(nan, posinf: Option<F>, neginf: Option<F>)` | `np.nan_to_num(a, nan, posinf, neginf)` | `None` keeps the default |
| `angle()`, `angle_deg()` | `np.angle(a)`, `np.angle(a, deg=True)` | real input: 0 or pi (`angle(-0.0) = pi`) |
| `unwrap()` | `np.unwrap(a)` | last axis, period `2 pi` |
| `unwrap_with(discont: Option<F>, axis: isize, period: F)` | `np.unwrap(a, discont, axis, period=period)` | floats only; panics on 0-d / bad axis |
| `ediff1d()`, `ediff1d_with(to_end: &[T], to_begin: &[T])` (`T: Num`) | `np.ediff1d` | flattens first |

## M2. Free functions at the crate root (broadcasting, scalars allowed anywhere)

`np::logaddexp(a, b)`, `np::logaddexp2`, `np::fmod`, `np::remainder` (NumPy `np.mod`; `mod` is a Rust
keyword so there is no alias), `np::floor_divide`, `np::divmod -> (q, r)`, `np::copysign`,
`np::heaviside(x, h0)`, `np::nextafter`, `np::ldexp(a, e: impl IntoArray<i32>)`,
`np::float_power -> Array<f64>` (always float64, like NumPy), `np::gcd`, `np::lcm` (`Array<i64>`,
non-negative results, `lcm` wraps on overflow like NumPy).

Windows (always `Array<f64>`): `np::hanning(m)`, `np::hamming(m)`, `np::blackman(m)`,
`np::bartlett(m)`, `np::kaiser(m, beta)`. `m = 0` is empty, `m = 1` is `[1.0]`.

Not wrapped: `np::real_if_close` (needs complex arrays, kernels `real_if_close_ok` /
`real_if_close_tol` are ready), integer `unwrap` (convert with `astype`).

## M3. NumPy behaviours the kernels reproduce

Checked value by value against NumPy 2.3.2 (f64 and f32 grids with NaN, +-inf, +-0.0, 5e-324, 1e300):

- `fmod` takes the dividend's sign; `remainder` / `floor_divide` / `divmod` use NumPy's `npy_divmod`
  fix-ups: `remainder(x, 0) = nan`, `remainder(-0.0, 3) = +0.0`, `remainder(5, -inf) = -inf`,
  `floor_divide(-1, inf) = -1`, `floor_divide(x, 0) = +-inf / nan`, correct sign of zero results.
- `logaddexp(inf, inf) = inf`, `logaddexp(-inf, -inf) = -inf`, `logaddexp(inf, -inf) = inf`, NaN propagates.
- `heaviside(nan, h) = nan`, `heaviside(0, h) = h`. `copysign(x, -0.0)` and `nextafter(-0.0, 0.0) = 0.0`,
  `nextafter(0.0, -1) = -5e-324`, `nextafter` with NaN gives NaN.
- `ldexp` is correctly rounded into the subnormal range (no double rounding) for exponents up to `i32::MIN/MAX`.
- `frexp` on subnormals normalizes first (exponent down to -1073).
- `rint` / `fix` / `negative` keep the sign of zero; `signbit(-nan)` is true.
- `unwrap`: boundary jumps of exactly `+-period/2` keep their sign, `discont` is not clamped (NumPy 2.3),
  NaN differences poison the running sum like `cumsum` does, `period=4` and `period=360` work.
- `gcd` / `lcm` for negative inputs, zeros, and `i64::MIN` (wraps like NumPy).
- `nan_to_num` defaults are `finfo.max` / `finfo.min` of the element type (f32 differs from f64).
- `angle(x)` of a real array uses imaginary part `+0`, so `angle(-0.0) = pi`.
- Windows follow NumPy's formulas and operation order (`pi * n / (M - 1)`); values agree to ~1e-15.

### Known differences from NumPy on this platform (all documented in the tests)

| Case | NumPy 2.3 (Windows) | rustnum | Reason |
|---|---|---|---|
| `frexp(inf)`, `frexp(nan)` exponent | -1 | 0 | C leaves it unspecified; glibc and musl return 0 |
| `exp/sinh/cosh(710+1j)` | `inf+infj` | finite, correct (`1.2e308+...`) | rustnum splits `exp(x)` to avoid intermediate overflow like C99 libms; verified with mpmath |
| `sqrt(1.7e308+1.7e308j)` imag part | half the right value | correct value | NumPy bug in its scaled branch; verified with mpmath |
| `(1e300+1e300j)**2` (`*`) | `-inf+infj` (FMA loop) | `nan+infj` | plain IEEE `ac - bd`; platform dependent in NumPy |
| Sign of zero / inf for non-finite inputs of `exp ln sin cos tan sinh cosh tanh powc` | C library specific | C99 Annex G (msun) | tests compare only the magnitude there |

## M4. `Complex<F>` (`misc/complex.rs`, `F = f32 | f64`)

```rust
pub struct Complex<F> { pub re: F, pub im: F }     // Clone Copy Debug Default PartialEq Display
```

- Constructors: `Complex::new(re, im)`, `from_real`, `from_polar(r, theta)`, `Complex::i()`.
- Operators: `+ - * /` and unary `-` between complex numbers, with a real scalar on either side
  (`z * 2.0`, `2.0 / z`), and `+= -= *= /=` with complex or real right-hand side. A real scalar is promoted to
  `s + 0j` first, exactly like NumPy, so `(inf + 1j) * 2.0` has a NaN imaginary part just as in NumPy.
  Division is Smith's algorithm (NumPy's loop), `x / 0j` gives complex inf/nan.
- Methods: `conj`, `norm` (`np.abs`, via `hypot`), `norm_sqr`, `arg` (`np.angle`), `exp`, `ln`, `sqrt`, `powc(w)`,
  `powf(p)`, `sin cos tan sinh cosh tanh`, `is_nan`, `is_infinite`, `is_finite`.
- Branch cuts follow the sign of a zero imaginary part: `sqrt(-4+0j) = 2j`, `sqrt(-4-0j) = -2j`,
  `ln(-1+0j) = pi j`, `ln(-1-0j) = -pi j`, `ln(0) = -inf + 0j`, `ln(-0.0+0j)` has imaginary part `pi`.
- `powc`: `z**0 = 1`, `0**w = 0` for `Re(w) > 0` else `nan+nanj`, integer exponents with `|n| < 100` by repeated
  squaring (the same multiplication order as NumPy), otherwise `exp(w * ln z)`.
- `Display` is `str(np.complex128(...))`: `(1+2j)`, `(-0.5-1j)`, `2j` when the real part is `+0`, `(1e+16+1j)`,
  `(inf+nanj)`; float32 switches to scientific notation at `1e6` like NumPy, float64 at `1e16`.
  Array printing (`1.+2.j`) arrives with the display change in M5.
- `cmp_numpy(&self, &other) -> Ordering`: NumPy's complex sort order (real part, then imaginary part, NaNs last:
  `[R+Rj, R+nanj, nan+Rj, nan+nanj]`), validated against `np.sort`.

## M5. Making `Complex<F>` an `Element` (v0.2 plan: what changes in the core)

`Complex<F>` is deliberately not an `Element` today because the v0.1 traits assume a totally ordered real
number. What has to change, in order:

1. **`Element` (`element.rs`).** Add `impl Element for Complex<f32>` / `Complex<f64>` with
   `DTYPE = "complex64" / "complex128"`, `to_f64 = re` and `from_f64(v) = v + 0j`. That matches NumPy's `astype`
   (real to complex sets `imag = 0`; complex to real drops the imaginary part). The `PartialOrd` supertrait of
   `Element` is the first blocker: either move it out of `Element` into a new `Ordered` trait, or implement
   `PartialOrd` for `Complex` lexicographically (NaN gives `None`). Moving it out is cleaner because sort,
   `maximum`, `min`, `max`, `argmax`, `clip`, comparisons and `unique` then say explicitly that they need an order.
2. **Split `Num`.** Today `Num` requires `Rem`, `abs() -> Self`, `MIN_VALUE`, `MAX_VALUE`. Complex has none of
   those (`abs` returns a real, `%` is undefined). Keep `Num` as the ring/field part (`+ - * /`, `ZERO`, `ONE`,
   `from_usize`) and add `Real: Num + Ordered + Rem` carrying `abs`, `MIN_VALUE`, `MAX_VALUE`. Sums, products,
   `cumsum`, `mean`, `matmul`, `dot`, `trace`, `reshape`, slicing and manipulation then work for complex arrays unchanged
   (they only need `Num`/`Element`).
3. **`ComplexFloat` trait** (`type Real: Float`, `re()`, `im()`, `norm()`, `conj()`, `exp`, `ln`, `sqrt`, `sin` ...
   all already implemented on `Complex<F>`): ufunc methods on `Array<Complex<F>>` mirror M1 of `API.md` section 7;
   `abs`, `angle`, `real`, `imag` return `Array<F>`; `conj`, `isnan`, `isfinite`, `isinf`, `real_if_close`,
   `iscomplex` are added; `var` / `std` return real arrays; `allclose` / `isclose` use `|a - b| <= atol + rtol * |b|`
   with the complex modulus.
4. **Ordering helpers.** A tiny `NumpyOrd { fn numpy_cmp(&self, &Self) -> Ordering }` implemented for reals
   (NaN last) and, via `Complex::cmp_numpy`, for complex; `sort`, `argsort`, `unique`, `searchsorted`,
   `min`, `max`, `argmin`, `argmax` use it, so complex arrays sort lexicographically like NumPy.
5. **Operators (`ops.rs`).** The scalar macro list gains `Complex<f32>` and `Complex<f64>` (`Array<Complex<F>> op F`
   and `Array<Complex<F>> op Complex<F>`). NumPy's implicit real-to-complex promotion (`Array<f64> + Array<Complex>`)
   is not provided: use `astype::<Complex<f64>>()` first (documented difference). `%` and the comparisons `< <= > >=`
   stay unavailable for complex, `==` / `!=` work.
6. **Display (`display.rs`).** One hook per element type for the cell text and its alignment width. Complex cells
   print as `1.+2.j`, `-0.5-1.j`, `nan+nanj` with the real and imaginary columns padded separately, using NumPy's
   precision-8 trimming per part; scalar `Display` already exists (M4).
7. **`io.rs`.** `.npy` descriptors `<c8` and `<c16` map to `Complex<f32>` and `Complex<f64>` (two interleaved floats
   per element, same endianness rules as the real types). CSV stays real-only.
8. **`linalg.rs`.** `matmul` / `dot` use the generic loop (`matrixmultiply` has no complex GEMM; a 4-real-GEMM
   split is the later optimisation). `det`, `inv`, `solve` become generic over `ComplexFloat` with pivoting on `norm()`;
   `eigh` (Hermitian) and `svd` follow after `.h()` (conjugate transpose) exists. Not part of the first complex release.
9. **Unify `KFloat` and `Float`.** Rename the methods of `KFloat` that clash with `Float` (or fold the two traits) so
   that `Float: KFloat` can be a supertrait without E0034 ambiguity. Then the wrappers in `api.rs` can bound on
   `Float` alone and generic user code can call them.

Estimated size: items 1-3 and 5 are mechanical (about a day), 6-7 are small, 8 is the only real project.
