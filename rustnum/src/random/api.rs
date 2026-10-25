//! Public random number generator API: `np::random::Rng`.

use super::engine::{self, Xoshiro256StarStar};
use crate::{Array, Element};

/// Random number generator using xoshiro256** seeded by SplitMix64, like
/// `np.random.default_rng`.
///
/// Same seed produces the same sequence on every platform. The streams differ from NumPy's
/// PCG64, so numbers match NumPy in distribution, not value.
///
/// Methods panic, with NumPy's wording, on invalid parameters (negative scale, `p` outside
/// `[0, 1]`, ...). The generator is cheap to clone; a clone continues the same stream.
///
/// # Examples
///
/// ```
/// use rustnum as np;
///
/// let mut rng = np::random::Rng::seed(42);
/// let a = rng.random(&[3, 4]);   // 3×4 of f64 in [0, 1)
/// assert_eq!(a.shape(), &[3, 4]);
///
/// let n = rng.normal(0.0, 1.0, &[1000]);
/// assert!(n.mean().abs() < 0.1);
/// ```
#[derive(Clone)]
pub struct Rng {
    inner: Xoshiro256StarStar,
}

impl Rng {
    /// Create a new RNG from a u64 seed, like `np.random.default_rng(seed)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.random(&[5]);
    /// assert_eq!(a.shape(), &[5]);
    /// ```
    pub fn seed(seed: u64) -> Self {
        Self {
            inner: Xoshiro256StarStar::seed(seed),
        }
    }

    /// Create a new RNG from system entropy (time + address).
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::from_entropy();
    /// let a = rng.random(&[5]);
    /// assert_eq!(a.shape(), &[5]);
    /// ```
    pub fn from_entropy() -> Self {
        Self {
            inner: Xoshiro256StarStar::from_entropy(),
        }
    }

    /// Uniform random f64 values in [0, 1), like `rng.random(shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(0);
    /// let a = rng.random(&[2, 3]);
    /// assert_eq!(a.shape(), &[2, 3]);
    /// for v in a.iter() {
    ///     assert!((0.0..1.0).contains(&v));
    /// }
    /// ```
    pub fn random(&mut self, shape: &[usize]) -> Array<f64> {
        let len: usize = shape.iter().product();
        let data = engine::random_vec(&mut self.inner, len);
        Array::from_vec(data, shape)
    }

    /// Uniform random f64 values in [low, high), like `rng.uniform(low, high, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(0);
    /// let a = rng.uniform(2.0, 5.0, &[100]);
    /// for v in a.iter() {
    ///     assert!(v >= 2.0 && v < 5.0);
    /// }
    /// ```
    pub fn uniform(&mut self, low: f64, high: f64, shape: &[usize]) -> Array<f64> {
        let len: usize = shape.iter().product();
        let data = engine::uniform_vec(&mut self.inner, low, high, len);
        Array::from_vec(data, shape)
    }

    /// Normal (Gaussian) random values, like `rng.normal(mean, std, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.normal(0.0, 1.0, &[10000]);
    /// let mean = a.mean();
    /// assert!((mean).abs() < 0.05, "mean={mean}");
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `std < 0` (`scale < 0`).
    pub fn normal(&mut self, mean: f64, std: f64, shape: &[usize]) -> Array<f64> {
        assert!(std >= 0.0, "scale < 0");
        let len: usize = shape.iter().product();
        let data = engine::normal_vec(&mut self.inner, mean, std, len);
        Array::from_vec(data, shape)
    }

    /// Standard normal (mean=0, std=1), like `rng.standard_normal(shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.standard_normal(&[5000]);
    /// assert!(a.mean().abs() < 0.05);
    /// ```
    pub fn standard_normal(&mut self, shape: &[usize]) -> Array<f64> {
        let len: usize = shape.iter().product();
        let data = engine::standard_normal_vec(&mut self.inner, len);
        Array::from_vec(data, shape)
    }

    /// Random integers in [low, high), like `rng.integers(low, high, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.integers(0, 10, &[100]);
    /// for v in a.iter() {
    ///     assert!(v >= 0 && v < 10);
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// Panics with `low >= high` if the range is empty.
    pub fn integers(&mut self, low: i64, high: i64, shape: &[usize]) -> Array<i64> {
        assert!(low < high, "low >= high");
        let len: usize = shape.iter().product();
        let data = engine::integers_vec(&mut self.inner, low, high, len);
        Array::from_vec(data, shape)
    }

    /// Draws `n` items along the first axis of `a`, like `rng.choice(a, n, replace)`.
    ///
    /// For a 1-d array the result has shape `[n]`; for higher dimensions whole sub-arrays
    /// are drawn and the result has shape `[n, ...]`. Without replacement every item is drawn at
    /// most once.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = np::arange(0.0, 10.0, 1.0);
    /// let chosen = rng.choice(&a, 5, false);
    /// assert_eq!(chosen.shape(), &[5]);
    /// let mut v = chosen.to_vec();
    /// v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    /// v.dedup();
    /// assert_eq!(v.len(), 5); // no repeats
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `a` is 0-d or empty while `n > 0`, or if `replace` is false and `n` exceeds
    /// the length of the first axis (`Cannot take a larger sample than population when
    /// replace is False`).
    pub fn choice<T: Element>(&mut self, a: &Array<T>, n: usize, replace: bool) -> Array<T> {
        assert!(a.ndim() >= 1, "a must be 1-dimensional or an integer");
        let pop = a.shape()[0];
        assert!(
            pop > 0 || n == 0,
            "a cannot be empty unless no samples are taken"
        );
        assert!(
            replace || n <= pop,
            "Cannot take a larger sample than population when replace is False"
        );
        let indices = engine::choice_indices(&mut self.inner, pop, n, replace);
        a.take(&indices, 0)
    }

    /// Shuffles `a` in place along its first axis, like `rng.shuffle(a)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let mut a = np::arange(0.0, 10.0, 1.0);
    /// rng.shuffle(&mut a);
    /// let mut v = a.to_vec();
    /// v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    /// assert_eq!(v, np::arange(0.0, 10.0, 1.0).to_vec());
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `a` is 0-d.
    pub fn shuffle<T: Element>(&mut self, a: &mut Array<T>) {
        assert!(
            a.ndim() >= 1,
            "shuffle requires an array with at least 1 dimension"
        );
        let order = engine::permutation_vec(&mut self.inner, a.shape()[0]);
        *a = a.take(&order, 0);
    }

    /// Random permutation of `0..n`, like `rng.permutation(n)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let p = rng.permutation(10);
    /// let mut sorted = p.to_vec();
    /// sorted.sort();
    /// assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    /// ```
    pub fn permutation(&mut self, n: usize) -> Array<usize> {
        Array::from_vec(engine::permutation_vec(&mut self.inner, n), &[n])
    }

    /// Binomial distribution, like `rng.binomial(n, p, shape)`; counts are `i64` as in NumPy.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.binomial(10, 0.5, &[1000]);
    /// assert!(a.iter().all(|k| (0..=10).contains(&k)));
    /// let mean = a.astype::<f64>().mean();
    /// assert!((mean - 5.0).abs() < 0.5);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `p` is outside `[0, 1]` or NaN, or if `n` does not fit an `i64`.
    pub fn binomial(&mut self, n: u64, p: f64, shape: &[usize]) -> Array<i64> {
        assert!((0.0..=1.0).contains(&p), "p < 0, p > 1 or p is NaN");
        assert!(i64::try_from(n).is_ok(), "n is too large");
        let len: usize = shape.iter().product();
        Array::from_vec(engine::binomial_vec(&mut self.inner, n, p, len), shape)
    }

    /// Poisson distribution, like `rng.poisson(lam, shape)`; counts are `i64` as in NumPy.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.poisson(5.0, &[1000]);
    /// let mean = a.astype::<f64>().mean();
    /// assert!((mean - 5.0).abs() < 0.5);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `lam` is negative, NaN or not finite.
    pub fn poisson(&mut self, lam: f64, shape: &[usize]) -> Array<i64> {
        assert!(lam >= 0.0, "lam < 0 or lam is NaN");
        assert!(lam.is_finite(), "lam value too large");
        let len: usize = shape.iter().product();
        Array::from_vec(engine::poisson_vec(&mut self.inner, lam, len), shape)
    }

    /// Exponential distribution with mean `scale`, like `rng.exponential(scale, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.exponential(2.0, &[10000]);
    /// assert!((a.mean() - 2.0).abs() < 0.1);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `scale < 0`.
    pub fn exponential(&mut self, scale: f64, shape: &[usize]) -> Array<f64> {
        assert!(scale >= 0.0, "scale < 0");
        let len: usize = shape.iter().product();
        Array::from_vec(engine::exponential_vec(&mut self.inner, scale, len), shape)
    }

    /// Beta distribution, like `rng.beta(a, b, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let arr = rng.beta(2.0, 5.0, &[10000]);
    /// // Expected mean = a / (a + b) = 2/7.
    /// assert!((arr.mean() - 2.0 / 7.0).abs() < 0.02);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics unless `a > 0` and `b > 0`.
    pub fn beta(&mut self, a: f64, b: f64, shape: &[usize]) -> Array<f64> {
        assert!(a > 0.0, "a <= 0");
        assert!(b > 0.0, "b <= 0");
        let len: usize = shape.iter().product();
        Array::from_vec(engine::beta_vec(&mut self.inner, a, b, len), shape)
    }

    /// Gamma distribution, like `rng.gamma(shape_param, scale, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let arr = rng.gamma(2.0, 1.0, &[10000]);
    /// // Expected mean = shape * scale = 2.
    /// assert!((arr.mean() - 2.0).abs() < 0.1);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `shape_param < 0` or `scale < 0`.
    pub fn gamma(&mut self, shape_param: f64, scale: f64, shape: &[usize]) -> Array<f64> {
        assert!(shape_param >= 0.0, "shape < 0");
        assert!(scale >= 0.0, "scale < 0");
        let len: usize = shape.iter().product();
        let data = engine::gamma_vec(&mut self.inner, shape_param, scale, len);
        Array::from_vec(data, shape)
    }
}
