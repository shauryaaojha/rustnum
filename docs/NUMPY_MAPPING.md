# NumPy to rustnum mapping

Left column: the NumPy call. Middle column: the rustnum call. Right column: what is different.

Conventions used in all tables:

- `use rustnum as np;` at the top of the file.
- `a` and `b` are `Array<f64>` unless the row says otherwise. `m` is an `Array<bool>`.
- `shape` is `&[usize]`, for example `&[2, 3]`.
- `axis` is `isize`. Negative values count from the end (`-1` is the last axis).
- Methods are called on the array (`a.sum()`). Free functions are called on the crate root (`np::zeros`) or modules (`np::linalg`, `np::random`, `np::io`, `np::fft`, `np::poly`).
- "Array-like" means any of `T`, `Array<T>`, `&Array<T>`, `Vec<T>`, `&[T]`, `[T; N]` (the `IntoArray<T>` trait).
- Anything not listed here does not exist in v0.1.

## Creation

| NumPy | rustnum | Notes |
|---|---|---|
| `np.array([1.0, 2.0, 3.0])` | `np::array![1.0, 2.0, 3.0]` | Macro. Integer literals give `Array<i32>`. Ragged input panics |
| `np.array([[1, 2], [3, 4]])` | `np::array![[1.0, 2.0], [3.0, 4.0]]` | 3-d nesting works too |
| `np.array(v).reshape(shape)` | `Array::from_vec(v, &[2, 3])` | Panics if `v.len()` is not the product of `shape` |
| `np.array(5.0)` (0-d) | `Array::scalar(5.0)` | |
| `np.array(list)` (1-d) | `Array::from(vec)` or `iter.collect::<Array<f64>>()` | `From<Vec<T>>` and `FromIterator<T>` |
| `np.fromfunction(f, shape)` | `Array::from_shape_fn(&[2, 3], \|idx\| ...)` | `idx` is `&[usize]` |
| `np.zeros((2, 3))` | `np::zeros::<f64>(&[2, 3])` | dtype is the type parameter |
| `np.ones((2, 3))` | `np::ones::<f64>(&[2, 3])` | |
| `np.full((2, 3), 7.0)` | `np::full(&[2, 3], 7.0)` | |
| `np.zeros_like(a)` | `np::zeros_like(&a)` | |
| `np.ones_like(a)` | `np::ones_like(&a)` | |
| `np.full_like(a, 7.0)` | `np::full_like(&a, 7.0)` | |
| `np.arange(0, 5, 1)` | `np::arange(0, 5, 1)` | All three arguments are required and have the same type. Use `0.0, 5.0, 1.0` for floats. Step 0 panics |
| `np.linspace(0, 1, 5)` | `np::linspace(0.0, 1.0, 5)` | Float only. Endpoint included |
| `np.logspace(0, 3, 4)` | `np::logspace(0.0, 3.0, 4)` | Base 10 |
| `np.eye(3)` | `np::eye::<f64>(3)` | |
| `np.identity(3)` | `np::identity::<f64>(3)` | |
| `np.diag(v)` / `np.diag(m)` | `np::diag(&v)` | 1-d gives a 2-d diagonal matrix; 2-d gives its diagonal as 1-d |
| `np.tri(3)` | `np::tri::<f64>(3)` | |
| `np.tril(a)` / `np.triu(a)` | `np::tril(&a)` / `np::triu(&a)` | |
| `X, Y = np.meshgrid(x, y)` | `let (xx, yy) = np::meshgrid(&x, &y);` | `indexing='xy'` only |
| `np.ascontiguousarray(a)` | `a.to_contiguous()` | No copy if already contiguous |
| `a.copy()` | `a.copy()` | Deep copy, contiguous |
| `a.astype(np.int64)` | `a.astype::<i64>()` | Casts like NumPy: floats truncate toward zero |
| `np.vectorize(f)(a)` | `a.map(\|x\| ...)` | Result element type can differ |
| in-place `f(a)` | `a.mapv_inplace(\|x\| ...)` | |
| `a.fill(v)` | `a.fill(v)` | |

## Attributes and element access

| NumPy | rustnum | Notes |
|---|---|---|
| `a.shape` | `a.shape()` | `&[usize]` |
| `a.ndim` | `a.ndim()` | |
| `a.size` | `a.size()` | |
| `len(a)` | `a.len()` | Length of axis 0. Panics on a 0-d array |
| `a.size == 0` | `a.is_empty()` | |
| `a.strides` | `a.strides()` | In elements, not bytes. `&[isize]` |
| `a.dtype` | `a.dtype()` | `&'static str` such as `"float64"` |
| `a.flags.c_contiguous` | `a.is_contiguous()` | |
| `a[1, 2]` | `a[[1, 2]]` or `a.get(&[1, 2])` | `get` returns the value. Out of bounds panics with `index 5 is out of bounds for axis 0 with size 3` |
| `a[1, 2] = 9.0` | `a[[1, 2]] = 9.0` or `a.set(&[1, 2], 9.0)` | Triggers copy-on-write if the buffer is shared |
| `a.item()` | `a.item()` | Panics unless `size == 1` |
| `a.tolist()` / `a.ravel().tolist()` | `a.to_vec()` | `Vec<T>` in C order |
| (no equivalent) | `a.as_slice()` | `Some(&[T])` only when contiguous |
| `a.flat` | `a.iter()` | Yields `T` in C order. `for x in &a` works too |
| `np.array_equal(a, b)` | `a.array_equal(&b)` | |
| `a == b` (whole array) | `a == b` | `PartialEq`: same shape and all elements equal. This returns one `bool` |
| `print(a)` | `println!("{a}")` | NumPy layout. `{a:?}` gives `array([...], dtype=float64)` |

## Indexing and slicing

| NumPy | rustnum | Notes |
|---|---|---|
| `a[1:3]` | `a.slice(&np::s![1..3])` | O(1) view (shares the buffer) |
| `a[1:3, :, -1]` | `a.slice(&np::s![1..3, .., -1])` | A single index removes that axis |
| `a[1:]`, `a[:3]`, `a[1:4]` | `s![1..]`, `s![..3]`, `s![1..4]` | |
| `a[1:4]` | `a.slice(&np::s![1..=3])` | Inclusive ranges work too: `1..=3` means the same as `1..4` |
| `a[::2, 1::-1]` | `a.slice(&np::s![..;2, 1..;-1])` | Step goes after `;` |
| `a[:, None]` | `a.expand_dims(1)` or `a.slice(&np::s![.., np::NEW_AXIS])` | Adds a new length-1 axis |
| `a[1:3] = v` | `a.slice_assign(&np::s![1..3], v)` | `v` is array-like and broadcasts. This is the explicit in-place write |
| `a[i]` (drop axis) | `a.index_axis(0, i)` | |
| `a[i]` on a 2-d array | `a.row(i)` | |
| `a[:, j]` on a 2-d array | `a.col(j)` | |
| `a[m]` (boolean mask) | `a.mask(&m)` | `m` must have the same shape as `a`. Result is 1-d |
| `a[m] = v` | `a.mask_assign(&m, v)` | `v` is a single `T` |
| `np.take(a, [0, 2], axis=0)` | `a.take(&[0, 2], 0)` | Indices are `&[usize]` |
| `np.compress(condition, a, axis=0)` | `a.compress(&condition, 0)` | `condition` is a 1-d `Array<bool>` |
| `np.nonzero(a)` | `np::nonzero(&a)` | `Vec<Array<usize>>`, one per axis |
| `np.argwhere(a)` | `np::argwhere(&a)` | `Array<usize>` |
| `np.where(m)` (one argument) | `np::nonzero(&m)` | |
| `np.searchsorted(sorted, v)` | `np::searchsorted(&sorted, v)` | Returns `usize` |

## Arithmetic, comparison and logic

| NumPy | rustnum | Notes |
|---|---|---|
| `a + b` | `&a + &b` | Broadcasts. `a + b` also works but moves both arrays |
| `a - b`, `a * b`, `a / b`, `a % b` | `&a - &b`, `&a * &b`, `&a / &b`, `&a % &b` | Same four borrow forms as `+` |
| `a + 1.0` | `&a + 1.0` | The scalar must have the array's element type |
| `1.0 - a` | `1.0 - &a` | Scalar on the left works for every numeric type |
| `-a` | `-&a` or `-a` | Signed and float types only |
| `a += b` | `a += &b` | Also `-= *= /= %=`. Right side broadcasts to the shape of `a`. Right side may be `Array`, `&Array` or `T` |
| `a == b` (element-wise) | `a.eq_elem(&b)` | Returns `Array<bool>`. The argument is array-like |
| `a != b` | `a.ne_elem(&b)` | |
| `a < b`, `a <= b` | `a.lt(&b)`, `a.le(&b)` | |
| `a > 0.5`, `a >= 0.5` | `a.gt(0.5)`, `a.ge(0.5)` | |
| `m1 & m2`, `m1 \| m2`, `m1 ^ m2`, `~m` | `&m1 & &m2`, `&m1 \| &m2`, `&m1 ^ &m2`, `!&m` | On `Array<bool>` |
| `np.logical_and(m1, m2)` | `np::logical_and(&m1, &m2)` | Also `logical_or`, `logical_xor`, `logical_not` |
| `np.allclose(a, b, rtol, atol)` | `a.allclose(&b, 1e-5, 1e-8)` | Float only. Both tolerances are required |
| `np.isclose(a, b, rtol, atol)` | `np::isclose(&a, &b, 1e-5, 1e-8)` | Returns `Array<bool>` |
| `np.broadcast_to(a, shape)` | `a.broadcast_to(&[3, 4])` | View with zero strides. Panics if not compatible |
| `np.broadcast_arrays(a, b)` | `np::broadcast_arrays(&a, &b)` | Returns a tuple of two arrays |
| (shape rule only) | `np::broadcast_shapes(&[2, 3], &[3])` | Returns `Option<Vec<usize>>` |
| `np.vectorize(f)(a, b)` | `a.zip_map(&b, \|x, y\| ...)` | Broadcasts |

## Element-wise math (ufuncs)

Most unary math functions are methods on `Array<F>` where `F` is `f32` or `f64`, returning a new array.
Binary math functions (`maximum`, `minimum`, `power`, `arctan2`, `hypot`, `where_`) are free functions in `np::`.

| NumPy | rustnum | Notes |
|---|---|---|
| `np.exp(a)`, `np.exp2(a)`, `np.expm1(a)` | `a.exp()`, `a.exp2()`, `a.expm1()` | |
| `np.log(a)`, `np.log2(a)`, `np.log10(a)`, `np.log1p(a)` | `a.log()`, `a.log2()`, `a.log10()`, `a.log1p()` | `log` is the natural log |
| `np.sqrt(a)`, `np.cbrt(a)`, `np.square(a)` | `a.sqrt()`, `a.cbrt()`, `a.square()` | `square` also works on integers |
| `np.sin(a)`, `np.cos(a)`, `np.tan(a)` | `a.sin()`, `a.cos()`, `a.tan()` | |
| `np.arcsin(a)`, `np.arccos(a)`, `np.arctan(a)` | `a.arcsin()`, `a.arccos()`, `a.arctan()` | |
| `np.sinh(a)`, `np.cosh(a)`, `np.tanh(a)` | `a.sinh()`, `a.cosh()`, `a.tanh()` | |
| `np.floor(a)`, `np.ceil(a)`, `np.trunc(a)` | `a.floor()`, `a.ceil()`, `a.trunc()` | |
| `np.round(a)` | `a.round()` | Bankers rounding to nearest integer |
| `np.round(a, decimals=2)` | `a.round_decimals(2)` | Round to `n` decimal places |
| `np.abs(a)` | `a.abs()` | Also on integers |
| `np.sign(a)` | `a.sign()` | Also on integers |
| `np.reciprocal(a)` | `a.reciprocal()` | |
| `np.power(a, 2.5)` | `a.powf(2.5)` | Scalar exponent |
| `a ** 2` | `a.powi(2)` | Integer exponent |
| `np.power(a, b)` | `np::power(&a, &b)` | Array exponent, broadcasts |
| `np.clip(a, lo, hi)` | `a.clip(lo, hi)` | Method only. Also on integers |
| `np.isnan(a)`, `np.isinf(a)`, `np.isfinite(a)` | `a.is_nan()`, `a.is_inf()`, `a.is_finite()` | Return `Array<bool>` |
| `np.maximum(a, b)` | `np::maximum(&a, &b)` | Broadcasts. Propagates NaN like NumPy. `np::maximum(&a, 0.0)` works |
| `np.minimum(a, b)` | `np::minimum(&a, &b)` | Same |
| `np.arctan2(y, x)` | `np::arctan2(&y, &x)` | |
| `np.hypot(a, b)` | `np::hypot(&a, &b)` | |
| `np.where(m, x, y)` | `np::where_(&m, &x, &y)` | `where` is a Rust keyword. `x` and `y` are array-like |
| `1 / (1 + np.exp(-a))` | `a.sigmoid()` | Extra, not in NumPy |
| `np.maximum(a, 0)` | `a.relu()` | Extra, not in NumPy |

## Miscellaneous math, special functions and windows

These element-wise functions, window generators and complex type live in the crate root.

| NumPy | rustnum | Notes |
|---|---|---|
| `np.sinc(a)` | `a.sinc()` | Normalised sinc: `sin(pi x) / (pi x)` |
| `np.i0(a)` | `a.i0()` | Modified Bessel function of the first kind, order 0 |
| `np.deg2rad(a)` | `a.deg2rad()` | Degrees to radians |
| `np.rad2deg(a)` | `a.rad2deg()` | Radians to degrees |
| `np.rint(a)` | `a.rint()` | Alias for `a.round()` |
| `np.fix(a)` | `a.fix()` | Alias for `a.trunc()` |
| `np.fabs(a)` | `a.fabs()` | Absolute value for floats |
| `np.positive(a)` | `a.positive()` | Returns a copy |
| `np.negative(a)` | `a.negative()` | Negation (`-&a`) |
| `np.signbit(a)` | `a.signbit()` | Returns `Array<bool>`, true where sign bit is set (including `-0.0`) |
| `np.spacing(a)` | `a.spacing()` | Distance to next representable float |
| `np.modf(a)` | `a.modf()` | Returns `(frac, int)` arrays |
| `np.frexp(a)` | `a.frexp()` | Returns `(mantissa, exponent)`: `(Array<F>, Array<i32>)` |
| `np.nan_to_num(a)` | `a.nan_to_num()` | Replace NaN with 0, inf with finite extremes |
| `np.nan_to_num(a, nan=..., posinf=..., neginf=...)` | `a.nan_to_num_with(nan, posinf, neginf)` | Custom replacement values (`None` keeps default) |
| `np.unwrap(a)` | `a.unwrap()` | Undo 2π jumps along last axis |
| `np.unwrap(a, discont=d, axis=ax, period=p)` | `a.unwrap_with(discont, axis, period)` | Custom discontinuity, axis and period |
| `np.ediff1d(a)` | `np::ediff1d(&a)` | Consecutive differences of flattened array |
| `np.ediff1d(a, to_end=e, to_begin=b)` | `np::ediff1d_with(&a, to_end, to_begin)` | With padding at begin/end |
| `np.angle(a)` | `np::angle(&a)` | Phase angle of real array (0 for x >= 0, π for x < 0) |
| `np.angle(a, deg=True)` | `np::angle_deg(&a)` | Phase angle in degrees |
| `np.logaddexp(a, b)` | `np::logaddexp(&a, &b)` | `log(exp(a) + exp(b))` without overflow |
| `np.logaddexp2(a, b)` | `np::logaddexp2(&a, &b)` | `log2(2**a + 2**b)` without overflow |
| `np.fmod(a, b)` | `np::fmod(&a, &b)` | C-style `fmod`, remainder has sign of dividend |
| `np.remainder(a, b)` / `np.mod(a, b)` | `np::remainder(&a, &b)` | Remainder has sign of divisor (Python `%`) |
| `np.floor_divide(a, b)` | `np::floor_divide(&a, &b)` | Floor division `a // b` |
| `np.divmod(a, b)` | `np::divmod(&a, &b)` | Returns `(floor_divide, remainder)` |
| `np.copysign(a, b)` | `np::copysign(&a, &b)` | Magnitude of `a` with sign of `b` |
| `np.heaviside(x, h0)` | `np::heaviside(&x, h0)` | Step function: 0 for x < 0, h0 at 0, 1 for x > 0 |
| `np.nextafter(a, b)` | `np::nextafter(&a, &b)` | Next float from `a` towards `b` |
| `np.ldexp(a, e)` | `np::ldexp(&a, &e)` | `a * 2**e`, `e` is `Array<i32>` or integer scalar |
| `np.float_power(a, b)` | `np::float_power(&a, &b)` | Always computed in `f64`, returns `Array<f64>` |
| `np.gcd(a, b)` | `np::gcd(&a, &b)` | Greatest common divisor for `i64` |
| `np.lcm(a, b)` | `np::lcm(&a, &b)` | Least common multiple for `i64` |
| `np.hanning(M)` | `np::hanning(m)` | Hann window of `m` points as `Array<f64>` |
| `np.hamming(M)` | `np::hamming(m)` | Hamming window of `m` points |
| `np.blackman(M)` | `np::blackman(m)` | Blackman window of `m` points |
| `np.bartlett(M)` | `np::bartlett(m)` | Bartlett (triangular) window of `m` points |
| `np.kaiser(M, beta)` | `np::kaiser(m, beta)` | Kaiser window with shape parameter `beta` |
| `np.complex128(re, im)` | `np::Complex::new(re, im)` | Scalar complex number type `Complex<F>` |

## Reductions

Each reduction over the whole array returns a plain value. Each `_axis` form removes that axis from the shape.
There is no `keepdims`: use `a.sum_axis(1).expand_dims(1)`.

| NumPy | rustnum | Notes |
|---|---|---|
| `a.sum()` | `a.sum()` | Returns `T`. Pairwise summation for accuracy |
| `a.sum(axis=1)` | `a.sum_axis(1)` | |
| `a.prod()`, `a.prod(axis=0)` | `a.prod()`, `a.prod_axis(0)` | |
| `a.min()`, `a.max()` | `a.min()`, `a.max()` | Empty array panics: `zero-size array to reduction operation minimum which has no identity` |
| `a.min(axis=0)`, `a.max(axis=0)` | `a.min_axis(0)`, `a.max_axis(0)` | |
| `a.mean()`, `a.mean(axis=0)` | `a.mean()`, `a.mean_axis(0)` | Float only |
| `a.var()`, `a.var(axis=0)` | `a.var()`, `a.var_axis(0)` | Float only. Population variance (`ddof=0`) like NumPy |
| `a.std()`, `a.std(axis=0)` | `a.std()`, `a.std_axis(0)` | Float only |
| `a.var(ddof=1)`, `a.std(ddof=1)` | `a.var_ddof(1)`, `a.std_ddof(1)` | Whole-array only |
| `a.var(axis=1, ddof=1)`, `a.std(axis=1, ddof=1)` | `a.var_ddof_axis(1, 1)`, `a.std_ddof_axis(1, 1)` | Along an axis with custom ddof |
| `a.argmin()`, `a.argmax()` | `a.argmin()`, `a.argmax()` | Return `usize`, an index into the flattened array |
| `a.argmin(axis=0)`, `a.argmax(axis=0)` | `a.argmin_axis(0)`, `a.argmax_axis(0)` | Return `Array<usize>` |
| `a.any()`, `a.all()` | `a.any()`, `a.all()` | Return `bool` |
| `a.any(axis=0)`, `a.all(axis=0)` | `a.any_axis(0)`, `a.all_axis(0)` | |
| `a.cumsum()`, `a.cumprod()` | `a.cumsum()`, `a.cumprod()` | Flat 1-d result, like NumPy with no axis |
| `a.cumsum(axis=1)`, `a.cumprod(axis=1)` | `a.cumsum_axis(1)`, `a.cumprod_axis(1)` | Keeps the shape |
| `np.count_nonzero(a)` | `a.count_nonzero()` | |
| `np.median(a)`, `np.median(a, axis=0)` | `a.median()`, `a.median_axis(0)` | Float only |
| `np.percentile(a, 90)` | `a.percentile(90.0)` | Float only. Whole-array only |
| `np.ptp(a)` | `a.ptp()` | Peak-to-peak difference (`max - min`) |
| `np.nansum(a)`, `np.nansum(a, axis=0)` | `a.nansum()`, `a.nansum_axis(0)` | Float only, treats NaN as 0 |
| `np.nanmean(a)`, `np.nanmean(a, axis=0)` | `a.nanmean()`, `a.nanmean_axis(0)` | Float only |
| `np.nanmin(a)`, `np.nanmin(a, axis=0)` | `a.nanmin()`, `a.nanmin_axis(0)` | Float only |
| `np.nanmax(a)`, `np.nanmax(a, axis=0)` | `a.nanmax()`, `a.nanmax_axis(0)` | Float only |

## Shape manipulation

| NumPy | rustnum | Notes |
|---|---|---|
| `a.reshape(2, 3)` | `a.reshape(&[2, 3])` | View if contiguous, copy otherwise. Wrong size panics: `cannot reshape array of size 6 into shape [4, 2]` |
| `a.reshape(-1, 3)` | `a.reshape_infer(&[-1, 3])` | `reshape` takes `usize`. Use `reshape_infer` (takes `isize`) when one entry is `-1` |
| `a.ravel()` | `a.ravel()` | |
| `a.flatten()` | `a.flatten()` | |
| `a.T` | `a.t()` | Reverses the axes. O(1) view |
| `a.transpose(1, 0, 2)` | `a.transpose(&[1, 0, 2])` | `permute_axes` is an alias |
| `np.swapaxes(a, 0, 1)` | `a.swapaxes(0, 1)` | |
| `np.moveaxis(a, 0, -1)` | `a.moveaxis(0, -1)` | |
| `np.squeeze(a)` | `a.squeeze()` | |
| `np.squeeze(a, axis=0)` | `a.squeeze_axis(0)` | |
| `np.expand_dims(a, 1)` | `a.expand_dims(1)` | |
| `np.flip(a, axis=0)` | `a.flip(0)` | View |
| `np.flip(a)` | `a.flip_all()` | |
| `np.roll(a, 2, axis=0)` | `a.roll(2, 0)` | Shift is `isize` |
| `np.repeat(a, 3, axis=0)` | `a.repeat(3, 0)` | |
| `np.tile(a, (2, 3))` | `a.tile(&[2, 3])` | |
| `np.sort(a)` | `a.sort()` | Along the last axis. NaN goes last |
| `np.sort(a, axis=0)` | `a.sort_axis(0)` | |
| `np.argsort(a)` | `a.argsort()` | Returns `Array<usize>` |
| `np.argsort(a, axis=0)` | `a.argsort_axis(0)` | |
| `np.concatenate([a, b], axis=0)` | `np::concatenate(&[&a, &b], 0)` | Takes a slice of references |
| `np.stack([a, b], axis=0)` | `np::stack(&[&a, &b], 0)` | |
| `np.vstack([a, b])` | `np::vstack(&[&a, &b])` | |
| `np.hstack([a, b])` | `np::hstack(&[&a, &b])` | |
| `np.split(a, 3, axis=0)` | `np::split(&a, 3, 0)` | Returns `Vec<Array<T>>`. Panics if not equal parts |
| `np.array_split(a, 3, axis=0)` | `np::array_split(&a, 3, 0)` | Unequal parts allowed |
| `np.unique(a)` | `np::unique(&a)` | Sorted 1-d array |
| `np.unique(a, return_counts=True)` | `np::unique_counts(&a)` | Returns `(values, counts)` |
| `np.pad(a, 1, constant_values=0)` | `np::pad(&a, 1, 0.0)` | Constant mode only, same width on all sides |

## Linear algebra

Matrix products, dot and trace are methods. Decompositions and solvers are free functions in `np::linalg`.

| NumPy | rustnum | Notes |
|---|---|---|
| `a @ b`, `np.matmul(a, b)` | `a.matmul(&b)` | 1-d, 2-d and batched n-d. Batch dimensions broadcast. f32 and f64 use rustnum's packed AVX2/FMA GEMM |
| `np.dot(a, b)` | `a.dot(&b)` | NumPy rules for up to 2-d |
| `np.trace(a)` | `a.trace()` | Trace along main diagonal |
| `np.outer(a, b)` | `np::linalg::outer(&a, &b)` | Outer product of flattened arrays |
| `np.inner(a, b)` | `np::linalg::inner(&a, &b)` | Inner product of two 1-d arrays |
| `np.vdot(a, b)` | `np::linalg::vdot(&a, &b)` | Dot product of flattened arrays |
| `np.linalg.det(a)` | `np::linalg::det(&a)` | Determinant of square matrix |
| `np.linalg.inv(a)` | `np::linalg::inv(&a)?` | `Result`. `Error::Singular` for singular matrix |
| `np.linalg.solve(a, b)` | `np::linalg::solve(&a, &b)?` | `Result`. Linear system solution |
| `np.linalg.lstsq(a, b)[0]` | `np::linalg::lstsq(&a, &b)?` | `Result`. Returns least-squares solution array |
| `np.linalg.pinv(a)` | `np::linalg::pinv(&a)` | Moore-Penrose pseudoinverse |
| `np.linalg.norm(a)` | `np::linalg::norm(&a)` | Frobenius norm for matrices, 2-norm for vectors |
| `np.linalg.norm(a, ord, axis)` | `np::linalg::norm_axis(&a, ord, axis)` | Vector norm along `axis` (`NormOrd`) |
| `np.linalg.vector_norm(x, ord=ord)` | `np::linalg::vector_norm(&x, ord)` | Norm of flattened vector (`NormOrd`) |
| `np.linalg.matrix_norm(a, ord=ord)` | `np::linalg::matrix_norm(&a, ord)` | Norm of 2-d matrix (`NormOrd`) |
| `np.linalg.cond(a)` | `np::linalg::cond(&a)` | 2-norm condition number `s_max / s_min` |
| `np.linalg.cond(a, p)` | `np::linalg::cond_ord(&a, ord)` | Condition number under `NormOrd` |
| `np.linalg.cholesky(a)` | `np::linalg::cholesky(&a)?` | Lower triangle. `Error::NotPositiveDefinite` |
| `np.linalg.qr(a)` | `np::linalg::qr(&a)` | Reduced mode. Returns `(q, r)` |
| `np.linalg.eigh(a)` | `np::linalg::eigh(&a)` | Ascending eigenvalues and eigenvectors as columns for symmetric input |
| `np.linalg.eigvalsh(a)` | `np::linalg::eigvalsh(&a)` | Ascending eigenvalues of symmetric matrix |
| `np.linalg.eig(a)` | `np::linalg::eig(&a)?` | `Result`. General square matrix: returns `(w_re, w_im, v_re, v_im)` |
| `np.linalg.eigvals(a)` | `np::linalg::eigvals(&a)?` | `Result`. Eigenvalues as `(re, im)` |
| `np.linalg.svd(a, full_matrices=False)` | `np::linalg::svd(&a)` | Returns `(u, s, vt)`. Reduced form. `s` descending |
| `np.linalg.slogdet(a)` | `np::linalg::slogdet(&a)` | Returns `(sign, logabsdet)` tuple |
| `np.linalg.matrix_rank(a)` | `np::linalg::matrix_rank(&a)` | Returns `usize` |
| `np.linalg.matrix_power(a, 3)` | `np::linalg::matrix_power(&a, 3)?` | `Result`. Negative power inverts first |
| `np.linalg.multi_dot([a, b, c])` | `np::linalg::multi_dot(&[&a, &b, &c])` | Cheapest matrix chain product |
| `np.tensordot(a, b, axes=(axes_a, axes_b))` | `np::linalg::tensordot(&a, &b, &axes_a, &axes_b)` | Tensor contraction over paired axes |
| `np.tensordot(a, b, axes=n)` | `np::linalg::tensordot_n(&a, &b, n)` | Contract last `n` of `a` with first `n` of `b` |

## Discrete Fourier Transform (np::fft)

Transforms over `Array<F>` (`f32` and `f64`). Since rustnum v0.1 has no complex array dtype, complex signals and spectra are pairs of real arrays `(re, im)` of the same shape.

| NumPy | rustnum | Notes |
|---|---|---|
| `np.fft.fft(a)` | `np::fft::fft(&re, &im)` | 1-d complex FFT along last axis, returns `(re, im)` |
| `np.fft.ifft(a)` | `np::fft::ifft(&re, &im)` | 1-d inverse complex FFT along last axis, divided by `n` |
| `np.fft.rfft(a)` | `np::fft::rfft(&a)` | Real FFT along last axis, returns `n / 2 + 1` bins as `(re, im)` |
| `np.fft.irfft(a, n)` | `np::fft::irfft(&re, &im, n)` | Inverse real FFT to length `n`, returns `Array<F>` |
| `np.fft.fft2(a)` | `np::fft::fft2(&re, &im)` | 2-d complex FFT over last two axes, returns `(re, im)` |
| `np.fft.ifft2(a)` | `np::fft::ifft2(&re, &im)` | 2-d inverse complex FFT over last two axes |
| `np.fft.fftfreq(n, d=1.0)` | `np::fft::fftfreq(n, d)` | Sample frequencies for `n`-point DFT with spacing `d` |
| `np.fft.rfftfreq(n, d=1.0)` | `np::fft::rfftfreq(n, d)` | Sample frequencies for `rfft`, `n / 2 + 1` bins |
| `np.fft.fftshift(a)` | `np::fft::fftshift(&a)` | Shift zero-frequency component to centre (all axes) |
| `np.fft.ifftshift(a)` | `np::fft::ifftshift(&a)` | Inverse of `fftshift` (all axes) |

## Polynomials (np::poly)

Polynomials with 1-d coefficients stored highest power first (`[1.0, -3.0, 2.0]` is `x^2 - 3x + 2`).

| NumPy | rustnum | Notes |
|---|---|---|
| `np.polyval(p, x)` | `np::poly::polyval(&p, &x)` | Evaluate polynomial at scalar or array `x` |
| `np.polyfit(x, y, deg)` | `np::poly::polyfit(&x, &y, deg)` | Least-squares polynomial fit of degree `deg` |
| `np.polyadd(a, b)` | `np::poly::polyadd(&a, &b)` | Sum of two polynomials |
| `np.polysub(a, b)` | `np::poly::polysub(&a, &b)` | Difference of two polynomials |
| `np.polymul(a, b)` | `np::poly::polymul(&a, &b)` | Product of two polynomials (convolution) |
| `np.polydiv(u, v)` | `np::poly::polydiv(&u, &v)` | Polynomial division, returns `(quotient, remainder)` |
| `np.polyder(p, m=1)` | `np::poly::polyder(&p, m)` | `m`-th derivative |
| `np.polyint(p, m=1, k=None)` | `np::poly::polyint(&p, m, &k)` | `m`-fold antiderivative with constants `k: &[F]` |
| `np.roots(p)` | `np::poly::roots(&p)` | Roots of polynomial, returns `(re, im)` arrays |

## Statistics, signal, sets and einsum

These functions live in the crate root (`rustnum::*`).

| NumPy | rustnum | Notes |
|---|---|---|
| `np.cov(m, bias=bias, ddof=ddof)` | `np::cov(&m, false, None)` | Covariance matrix of rows (variables). Output is `Array<f64>` |
| `np.corrcoef(x)` | `np::corrcoef(&x)` | Pearson correlation matrix of rows |
| `np.histogram(a, bins, range, density, weights)` | `np::histogram(&a, bins, range, density, weights)` | Returns `(counts, edges)` as `Array<f64>`. `bins` converts from `usize`, `Vec<f64>`, `&[f64]` |
| `np.histogram_bin_edges(a, bins, range)` | `np::histogram_bin_edges(&a, bins, range)` | Returns bin edges `Array<f64>` |
| `np.bincount(x, weights, minlength)` | `np::bincount(&x, weights, minlength)` | Count occurrences of non-negative `i64`. Output is `Array<f64>` |
| `np.digitize(x, bins, right=False)` | `np::digitize(&x, &bins, right)` | Bin index for each value, returns `Array<usize>` |
| `np.average(a, weights=weights)` | `np::average(&a, weights)` | Weighted average of whole array |
| `np.average(a, axis=axis, weights=weights)` | `np::average_axis(&a, weights, axis)` | Weighted average along `axis` |
| `np.quantile(a, q)` | `np::quantile(&a, q)` | Linear quantile for `q` in `[0, 1]` |
| `np.quantile(a, q, axis=axis)` | `np::quantile_axis(&a, q, axis)` | Quantile along an axis |
| `np.nanquantile(a, q)` | `np::nanquantile(&a, q)` | NaN-ignoring quantile |
| `np.nanquantile(a, q, axis=axis)` | `np::nanquantile_axis(&a, q, axis)` | NaN-ignoring quantile along an axis |
| `np.convolve(a, v, mode="full")` | `np::convolve(&a, &v, np::ConvolveMode::Full)` | 1-d discrete convolution (`Full`, `Same`, `Valid`) |
| `np.correlate(a, v, mode="valid")` | `np::correlate(&a, &v, np::ConvolveMode::Valid)` | 1-d cross-correlation |
| `np.diff(a, n=1, axis=-1)` | `np::diff(&a, n, axis)` | `n`-th discrete difference along `axis` |
| `np.gradient(f, dx)` | `np::gradient(&f, dx)` | Central differences; returns `Vec<Array<F>>`, one per axis |
| `np.gradient(f, spacing, axis=axis, edge_order=1)` | `np::gradient_axis(&f, spacing, axis, edge_order)` | Gradient along one axis (`Spacing::Uniform` or `Spacing::Coords`) |
| `np.interp(x, xp, fp, left, right)` | `np::interp(&x, &xp, &fp, left, right)` | 1-d linear interpolation. `left`/`right` default to `fp[0]` and `fp[last]` |
| `np.trapezoid(y, x_or_dx, axis=-1)` | `np::trapezoid(&y, spacing, axis)` | Trapezoidal rule along `axis` |
| `np.cross(a, b)` | `np::cross(&a, &b)` | Cross product of 3-vectors or 2-vectors along last axis |
| `np.kron(a, b)` | `np::kron(&a, &b)` | Kronecker product |
| `np.intersect1d(a, b)` | `np::intersect1d(&a, &b)` | Sorted unique values in both arrays |
| `np.union1d(a, b)` | `np::union1d(&a, &b)` | Sorted unique values in either array |
| `np.setdiff1d(a, b)` | `np::setdiff1d(&a, &b)` | Sorted unique values in `a` not in `b` |
| `np.setxor1d(a, b)` | `np::setxor1d(&a, &b)` | Sorted unique values in exactly one array |
| `np.isin(element, test_elements)` | `np::isin(&element, &test_elements)` | Returns `Array<bool>` with shape of `element` |
| `np.einsum(subscripts, *operands)` | `np::einsum("ij,jk->ik", &[&a, &b])` | Einstein summation with explicit/implicit output, ellipsis `...`, diagonals |

## Random

Create a generator once, then call methods on it. Every method needs `&mut`, so declare it with `let mut rng`.

| NumPy | rustnum | Notes |
|---|---|---|
| `rng = np.random.default_rng(42)` | `let mut rng = np::random::Rng::seed(42);` | xoshiro256**, not PCG64 |
| (seed from OS) | `np::random::Rng::from_entropy()` | |
| `rng.random((2, 3))` | `rng.random(&[2, 3])` | Uniform in [0, 1). `Array<f64>` |
| `rng.uniform(0, 1, (2, 3))` | `rng.uniform(0.0, 1.0, &[2, 3])` | |
| `rng.normal(0, 1, (2, 3))` | `rng.normal(0.0, 1.0, &[2, 3])` | |
| `rng.standard_normal((2, 3))` | `rng.standard_normal(&[2, 3])` | |
| `rng.integers(0, 10, (2, 3))` | `rng.integers(0, 10, &[2, 3])` | `Array<i64>`. High is exclusive |
| `rng.choice(a, n, replace=False)` | `rng.choice(&a, n, false)` | |
| `rng.shuffle(a)` | `rng.shuffle(&mut a)` | Along the first axis |
| `rng.permutation(n)` | `rng.permutation(n)` | `Array<usize>` |
| `rng.binomial(n, p, size)` | `rng.binomial(n, p, &[2, 3])` | |
| `rng.poisson(lam, size)` | `rng.poisson(lam, &[2, 3])` | |
| `rng.exponential(scale, size)` | `rng.exponential(scale, &[2, 3])` | |

The same seed gives the same numbers on every platform, but they are **not** the numbers NumPy gives for the same seed.

## I/O

All I/O functions live in `np::io` and return `Result`.

| NumPy | rustnum | Notes |
|---|---|---|
| `np.save("a.npy", a)` | `np::io::save_npy("a.npy", &a)?` | `Result`. Format v1.0, little-endian |
| `np.load("a.npy")` | `np::io::load_npy::<f64>("a.npy")?` | The type must match the file, else `Error::DType`. v1.0 and v2.0, C and Fortran order |
| `np.savez("a.npz", a=a, b=b)` | `np::io::savez("a.npz", &[("a", &a), ("b", &b)])?` | Uncompressed ZIP archive of `.npy` arrays |
| `np.savez_compressed("a.npz", a=a, b=b)` | `np::io::savez_compressed("a.npz", &[("a", &a), ("b", &b)])?` | DEFLATE-compressed `.npz` archive |
| `dict(np.load("a.npz"))` | `np::io::load_npz::<f64>("a.npz")?` | Loads all members into `Vec<(String, Array<T>)>` |
| `f = np.load("a.npz")` | `let f = np::io::NpzFile::open("a.npz")?;` | Lazy archive reader: `f.keys()`, `f.get::<f64>("a")?` |
| `np.savetxt("a.csv", a, delimiter=",")` | `np::io::save_csv("a.csv", &a, ',')?` | 2-d arrays only |
| `np.loadtxt("a.csv", delimiter=",", skiprows=1)` | `np::io::load_csv::<f64>("a.csv", ',', 1)?` | |
| (bytes in memory) | `np::io::to_npy_bytes(&a)`, `np::io::from_npy_bytes::<f64>(&bytes)?` | In-memory `.npy` codec |

## Common gotchas

**1. Views are copy-on-write, not write-through.** In NumPy, changing a slice changes the parent array. In rustnum, a slice
shares memory until someone writes to it; the writer then gets its own private copy.

```rust
let a = np::zeros::<f64>(&[2, 3]);
let mut row = a.row(0);
row.fill(9.0);                      // `a` is unchanged
let mut b = a.copy();
b.slice_assign(&np::s![0, ..], 9.0); // to write into part of an array, say so
```

**2. No keyword arguments.** `a.sum(axis=1)` is `a.sum_axis(1)`. `keepdims=True` is `.expand_dims(axis)` after the reduction.
`ddof=1` is `var_ddof(1)` / `std_ddof(1)` (or `var_ddof_axis(axis, 1)` / `std_ddof_axis(axis, 1)`).

**3. `where` is `where_`.** `where` is a Rust keyword. `np.where(m, x, y)` is `np::where_(&m, x, y)`.

**4. `-1` in reshape needs `reshape_infer`.** `a.reshape(&[2, 3])` takes `usize`. `a.reshape_infer(&[-1, 3])` takes `isize`.

**5. Float literals need `.0`.** Rust does not turn `5` into `5.0`.

```rust
let x = np::arange(0.0, 5.0, 1.0);  // right: Array<f64>
let y = np::arange(0, 5, 1);        // Array<i32>, not f64
let z = &x * 2.0;                   // right
// let z = &x * 2;                  // compile error: expected f64, found integer
```

`np::array![1, 2, 3]` is an `Array<i32>`. Write `np::array![1.0, 2.0, 3.0]` for floats, or convert with `.astype::<f64>()`.

**6. `&a + &b` borrows, `a + b` moves.** After `let c = a + b;` you cannot use `a` or `b` again. Arrays are cheap to clone
(`a.clone()` does not copy data), but borrowing is cleaner: `let c = &a + &b;`.

**7. Scalars work on both sides.** `&a * 2.0` and `2.0 * &a` are both fine.

**8. The scalar type must match the array.** `Array<f32> * 2.0_f64` does not compile. Use `Array<f32> * 2.0_f32`.

**9. Comparisons are methods.** `a == b` returns one `bool` (same shape and values). For an element-wise result use
`a.eq_elem(&b)`, `a.gt(0.5)` and so on.

**10. Misuse panics, bad data returns `Result`.** Shape mismatches, bad indices and bad axes panic with NumPy's message.
File errors, parse errors, singular matrices and non-positive-definite matrices return `Err`. Put `?` after
`np::linalg::solve`, `inv`, `cholesky`, `lstsq`, `matrix_power`, `eig`, `eigvals` and all `np::io` functions.

**11. Random streams differ from NumPy.** `Rng::seed(42)` is reproducible in rustnum but does not match
`np.random.default_rng(42)`. Do not compare random numbers across the two. Compare statistics instead.

**12. `argmax` is a flat index.** `a.argmax()` indexes the flattened array. Use `a.argmax_axis(axis)` for a per-axis result.

**13. `eigh` returns ascending eigenvalues.** For PCA you usually want the largest first: `vals.flip(0)`, `vecs.flip(1)`.

**14. A mask must match the array shape.** `a.mask(&m)` needs `m.shape() == a.shape()`. To select rows by a 1-d condition,
use `a.take(&np::nonzero(&cond)[0].to_vec(), 0)` or `a.compress(&cond, 0)`.

**15. Complex data is represented as `(re, im)` pairs.** `Array` does not yet have a complex element type; `np::fft::fft`,
`np::poly::roots`, and `np::linalg::eig` take or return pairs of real arrays `(re, im)`. Scalar complex arithmetic is available via `np::Complex<F>`.

## More NumPy functions

Added after v0.1. Same conventions as above.

### NaN-aware reductions

| NumPy | rustnum | Notes |
|---|---|---|
| `np.nanvar(a)`, `np.nanvar(a, axis=0)` | `a.nanvar()`, `a.nanvar_axis(0)` | Float only. `ddof`: `a.nanvar_ddof(1)`, `a.nanvar_ddof_axis(0, 1)` |
| `np.nanstd(a)`, `np.nanstd(a, axis=0)` | `a.nanstd()`, `a.nanstd_axis(0)` | Float only. `ddof`: `a.nanstd_ddof(1)`, `a.nanstd_ddof_axis(0, 1)` |
| `np.nanmedian(a)`, `np.nanmedian(a, axis=0)` | `a.nanmedian()`, `a.nanmedian_axis(0)` | All-NaN lane gives NaN |
| `np.nanpercentile(a, q)`, `np.nanpercentile(a, q, axis=0)` | `a.nanpercentile(q)`, `a.nanpercentile_axis(q, 0)` | `q` in 0..=100, else panics `Percentiles must be in the range [0, 100]` |
| `np.nanargmax(a)`, `np.nanargmin(a)` | `a.nanargmax()`, `a.nanargmin()` | Flat index. All-NaN input panics `All-NaN slice encountered` |
| `np.nanargmax(a, axis=0)`, `np.nanargmin(a, axis=0)` | `a.nanargmax_axis(0)`, `a.nanargmin_axis(0)` | `Array<usize>` |
| `np.nanprod(a)`, `np.nanprod(a, axis=0)` | `a.nanprod()`, `a.nanprod_axis(0)` | NaN counts as 1 |
| `np.nancumsum(a)`, `np.nancumsum(a, axis=0)` | `a.nancumsum()`, `a.nancumsum_axis(0)` | NaN counts as 0; flat result without axis |
| `np.nancumprod(a)`, `np.nancumprod(a, axis=0)` | `a.nancumprod()`, `a.nancumprod_axis(0)` | NaN counts as 1; flat result without axis |
| `np.nanquantile` | `np::nanquantile`, `np::nanquantile_axis` | Already existed (see Statistics section) |

## Not in v0.1

`float16`, structured and datetime dtypes, masked arrays, memory-mapped files, string arrays,
multidimensional fancy indexing with index arrays on several axes at once, and `Array<Complex<F>>` element type.
