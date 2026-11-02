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

    /// Standard exponential distribution, like `rng.standard_exponential(shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.standard_exponential(&[1000]);
    /// assert!((a.mean() - 1.0).abs() < 0.1);
    /// ```
    pub fn standard_exponential(&mut self, shape: &[usize]) -> Array<f64> {
        let len: usize = shape.iter().product();
        Array::from_vec(
            engine::standard_exponential_vec(&mut self.inner, len),
            shape,
        )
    }

    /// Standard gamma distribution, like `rng.standard_gamma(shape_param, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.standard_gamma(2.0, &[1000]);
    /// assert!((a.mean() - 2.0).abs() < 0.15);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `shape_param < 0`.
    pub fn standard_gamma(&mut self, shape_param: f64, shape: &[usize]) -> Array<f64> {
        assert!(shape_param >= 0.0, "shape < 0");
        let len: usize = shape.iter().product();
        Array::from_vec(
            engine::standard_gamma_vec(&mut self.inner, shape_param, len),
            shape,
        )
    }

    /// Standard Cauchy distribution, like `rng.standard_cauchy(shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.standard_cauchy(&[50]);
    /// assert_eq!(a.shape(), &[50]);
    /// ```
    pub fn standard_cauchy(&mut self, shape: &[usize]) -> Array<f64> {
        let len: usize = shape.iter().product();
        Array::from_vec(engine::standard_cauchy_vec(&mut self.inner, len), shape)
    }

    /// Student's t distribution, like `rng.standard_t(df, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.standard_t(10.0, &[1000]);
    /// assert!(a.mean().abs() < 0.2);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `df <= 0`.
    pub fn standard_t(&mut self, df: f64, shape: &[usize]) -> Array<f64> {
        assert!(df > 0.0, "df <= 0");
        let len: usize = shape.iter().product();
        Array::from_vec(engine::standard_t_vec(&mut self.inner, df, len), shape)
    }

    /// Chi-square distribution, like `rng.chisquare(df, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.chisquare(4.0, &[1000]);
    /// assert!((a.mean() - 4.0).abs() < 0.3);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `df <= 0`.
    pub fn chisquare(&mut self, df: f64, shape: &[usize]) -> Array<f64> {
        assert!(df > 0.0, "df <= 0");
        let len: usize = shape.iter().product();
        Array::from_vec(engine::chisquare_vec(&mut self.inner, df, len), shape)
    }

    /// Noncentral chi-square distribution, like `rng.noncentral_chisquare(df, nonc, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.noncentral_chisquare(4.0, 2.0, &[1000]);
    /// assert!((a.mean() - 6.0).abs() < 0.5);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `df <= 0` or `nonc < 0`.
    pub fn noncentral_chisquare(&mut self, df: f64, nonc: f64, shape: &[usize]) -> Array<f64> {
        assert!(df > 0.0, "df <= 0");
        assert!(nonc >= 0.0, "nonc < 0");
        let len: usize = shape.iter().product();
        Array::from_vec(
            engine::noncentral_chisquare_vec(&mut self.inner, df, nonc, len),
            shape,
        )
    }

    /// Fisher-Snedecor F distribution, like `rng.f(dfnum, dfden, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.f(5.0, 10.0, &[1000]);
    /// assert!((a.mean() - 1.25).abs() < 0.2);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `dfnum <= 0` or `dfden <= 0`.
    pub fn f(&mut self, dfnum: f64, dfden: f64, shape: &[usize]) -> Array<f64> {
        assert!(dfnum > 0.0, "dfnum <= 0");
        assert!(dfden > 0.0, "dfden <= 0");
        let len: usize = shape.iter().product();
        Array::from_vec(engine::f_vec(&mut self.inner, dfnum, dfden, len), shape)
    }

    /// Noncentral F distribution, like `rng.noncentral_f(dfnum, dfden, nonc, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.noncentral_f(5.0, 10.0, 1.0, &[1000]);
    /// assert_eq!(a.shape(), &[1000]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `dfnum <= 0`, `dfden <= 0`, or `nonc < 0`.
    pub fn noncentral_f(
        &mut self,
        dfnum: f64,
        dfden: f64,
        nonc: f64,
        shape: &[usize],
    ) -> Array<f64> {
        assert!(dfnum > 0.0, "dfnum <= 0");
        assert!(dfden > 0.0, "dfden <= 0");
        assert!(nonc >= 0.0, "nonc < 0");
        let len: usize = shape.iter().product();
        Array::from_vec(
            engine::noncentral_f_vec(&mut self.inner, dfnum, dfden, nonc, len),
            shape,
        )
    }

    /// Laplace distribution, like `rng.laplace(loc, scale, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.laplace(2.0, 1.0, &[1000]);
    /// assert!((a.mean() - 2.0).abs() < 0.2);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `scale < 0`.
    pub fn laplace(&mut self, loc: f64, scale: f64, shape: &[usize]) -> Array<f64> {
        assert!(scale >= 0.0, "scale < 0");
        let len: usize = shape.iter().product();
        Array::from_vec(engine::laplace_vec(&mut self.inner, loc, scale, len), shape)
    }

    /// Logistic distribution, like `rng.logistic(loc, scale, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.logistic(3.0, 2.0, &[1000]);
    /// assert!((a.mean() - 3.0).abs() < 0.3);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `scale < 0`.
    pub fn logistic(&mut self, loc: f64, scale: f64, shape: &[usize]) -> Array<f64> {
        assert!(scale >= 0.0, "scale < 0");
        let len: usize = shape.iter().product();
        Array::from_vec(
            engine::logistic_vec(&mut self.inner, loc, scale, len),
            shape,
        )
    }

    /// Lognormal distribution, like `rng.lognormal(mean, sigma, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.lognormal(0.0, 1.0, &[1000]);
    /// assert!((a.mean() - 1.65).abs() < 0.3);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `sigma < 0`.
    pub fn lognormal(&mut self, mean: f64, sigma: f64, shape: &[usize]) -> Array<f64> {
        assert!(sigma >= 0.0, "sigma < 0");
        let len: usize = shape.iter().product();
        Array::from_vec(
            engine::lognormal_vec(&mut self.inner, mean, sigma, len),
            shape,
        )
    }

    /// Gumbel distribution, like `rng.gumbel(loc, scale, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.gumbel(0.0, 1.0, &[1000]);
    /// assert!((a.mean() - 0.577).abs() < 0.2);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `scale < 0`.
    pub fn gumbel(&mut self, loc: f64, scale: f64, shape: &[usize]) -> Array<f64> {
        assert!(scale >= 0.0, "scale < 0");
        let len: usize = shape.iter().product();
        Array::from_vec(engine::gumbel_vec(&mut self.inner, loc, scale, len), shape)
    }

    /// Rayleigh distribution, like `rng.rayleigh(scale, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.rayleigh(2.0, &[1000]);
    /// assert!((a.mean() - 2.0 * (std::f64::consts::PI / 2.0).sqrt()).abs() < 0.2);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `scale < 0`.
    pub fn rayleigh(&mut self, scale: f64, shape: &[usize]) -> Array<f64> {
        assert!(scale >= 0.0, "scale < 0");
        let len: usize = shape.iter().product();
        Array::from_vec(engine::rayleigh_vec(&mut self.inner, scale, len), shape)
    }

    /// Pareto distribution (Pareto II / Lomax), like `rng.pareto(a, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let arr = rng.pareto(3.0, &[1000]);
    /// assert!((arr.mean() - 0.5).abs() < 0.1);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `a <= 0`.
    pub fn pareto(&mut self, a: f64, shape: &[usize]) -> Array<f64> {
        assert!(a > 0.0, "a <= 0");
        let len: usize = shape.iter().product();
        Array::from_vec(engine::pareto_vec(&mut self.inner, a, len), shape)
    }

    /// Power distribution, like `rng.power(a, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.power(2.0, &[1000]);
    /// assert!((a.mean() - 2.0 / 3.0).abs() < 0.1);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `a <= 0`.
    pub fn power(&mut self, a: f64, shape: &[usize]) -> Array<f64> {
        assert!(a > 0.0, "a <= 0");
        let len: usize = shape.iter().product();
        Array::from_vec(engine::power_vec(&mut self.inner, a, len), shape)
    }

    /// Weibull distribution, like `rng.weibull(a, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.weibull(2.0, &[1000]);
    /// assert_eq!(a.shape(), &[1000]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `a < 0`.
    pub fn weibull(&mut self, a: f64, shape: &[usize]) -> Array<f64> {
        assert!(a >= 0.0, "a < 0");
        let len: usize = shape.iter().product();
        Array::from_vec(engine::weibull_vec(&mut self.inner, a, len), shape)
    }

    /// Wald (Inverse Gaussian) distribution, like `rng.wald(mean, scale, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.wald(2.0, 3.0, &[1000]);
    /// assert!((a.mean() - 2.0).abs() < 0.2);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `mean <= 0` or `scale <= 0`.
    pub fn wald(&mut self, mean: f64, scale: f64, shape: &[usize]) -> Array<f64> {
        assert!(mean > 0.0, "mean <= 0");
        assert!(scale > 0.0, "scale <= 0");
        let len: usize = shape.iter().product();
        Array::from_vec(engine::wald_vec(&mut self.inner, mean, scale, len), shape)
    }

    /// Triangular distribution, like `rng.triangular(left, mode, right, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.triangular(0.0, 1.0, 3.0, &[1000]);
    /// assert!((a.mean() - 4.0 / 3.0).abs() < 0.2);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `left > mode`, `mode > right`, or `left == right`.
    pub fn triangular(&mut self, left: f64, mode: f64, right: f64, shape: &[usize]) -> Array<f64> {
        assert!(left <= mode, "left > mode");
        assert!(mode <= right, "mode > right");
        assert!(left < right, "left == right");
        let len: usize = shape.iter().product();
        Array::from_vec(
            engine::triangular_vec(&mut self.inner, left, mode, right, len),
            shape,
        )
    }

    /// Von Mises distribution, like `rng.vonmises(mu, kappa, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.vonmises(0.0, 2.0, &[1000]);
    /// assert_eq!(a.shape(), &[1000]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `kappa < 0`.
    pub fn vonmises(&mut self, mu: f64, kappa: f64, shape: &[usize]) -> Array<f64> {
        assert!(kappa >= 0.0, "kappa < 0");
        let len: usize = shape.iter().product();
        Array::from_vec(engine::vonmises_vec(&mut self.inner, mu, kappa, len), shape)
    }

    /// Geometric distribution, like `rng.geometric(p, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.geometric(0.5, &[1000]);
    /// assert!((a.astype::<f64>().mean() - 2.0).abs() < 0.2);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `p <= 0` or `p > 1`.
    pub fn geometric(&mut self, p: f64, shape: &[usize]) -> Array<i64> {
        assert!(p > 0.0 && p <= 1.0, "p <= 0, p > 1 or p contains NaNs");
        let len: usize = shape.iter().product();
        Array::from_vec(engine::geometric_vec(&mut self.inner, p, len), shape)
    }

    /// Negative binomial distribution, like `rng.negative_binomial(n, p, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.negative_binomial(5.0, 0.5, &[1000]);
    /// assert!((a.astype::<f64>().mean() - 5.0).abs() < 0.5);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `n <= 0` or `p <= 0` or `p > 1`.
    pub fn negative_binomial(&mut self, n: f64, p: f64, shape: &[usize]) -> Array<i64> {
        assert!(n > 0.0, "n <= 0");
        assert!(p > 0.0 && p <= 1.0, "p <= 0, p > 1 or p contains NaNs");
        let len: usize = shape.iter().product();
        Array::from_vec(
            engine::negative_binomial_vec(&mut self.inner, n, p, len),
            shape,
        )
    }

    /// Hypergeometric distribution, like `rng.hypergeometric(ngood, nbad, nsample, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.hypergeometric(100, 100, 20, &[1000]);
    /// assert!((a.astype::<f64>().mean() - 10.0).abs() < 0.5);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `ngood < 0`, `nbad < 0`, `nsample < 0`, or `nsample > ngood + nbad`.
    pub fn hypergeometric(
        &mut self,
        ngood: i64,
        nbad: i64,
        nsample: i64,
        shape: &[usize],
    ) -> Array<i64> {
        assert!(ngood >= 0, "ngood < 0");
        assert!(nbad >= 0, "nbad < 0");
        assert!(nsample >= 0, "nsample < 0");
        assert!(nsample <= ngood + nbad, "ngood + nbad < nsample");
        let len: usize = shape.iter().product();
        Array::from_vec(
            engine::hypergeometric_vec(&mut self.inner, ngood, nbad, nsample, len),
            shape,
        )
    }

    /// Multivariate hypergeometric distribution, like `rng.multivariate_hypergeometric(colors, nsample, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.multivariate_hypergeometric(&[10, 20, 30], 12, &[5]);
    /// assert_eq!(a.shape(), &[5, 3]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `colors` has negative elements, `nsample < 0`, or `nsample > sum(colors)`.
    pub fn multivariate_hypergeometric(
        &mut self,
        colors: &[i64],
        nsample: i64,
        shape: &[usize],
    ) -> Array<i64> {
        assert!(
            colors.iter().all(|&c| c >= 0),
            "colors must be a one-dimensional sequence of nonnegative integers not exceeding 9223372036854775807."
        );
        assert!(nsample >= 0, "nsample must be nonnegative.");
        let total: i64 = colors.iter().sum();
        assert!(nsample <= total, "nsample > sum(colors)");
        let k = colors.len();
        let count: usize = shape.iter().product();
        let actual_count = if shape.is_empty() { 1 } else { count };
        let data =
            engine::multivariate_hypergeometric_vec(&mut self.inner, colors, nsample, actual_count);
        let out_shape = if shape.is_empty() {
            vec![k]
        } else {
            let mut s = shape.to_vec();
            s.push(k);
            s
        };
        Array::from_vec(data, &out_shape)
    }

    /// Logarithmic series distribution, like `rng.logseries(p, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.logseries(0.6, &[1000]);
    /// assert_eq!(a.shape(), &[1000]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `p < 0` or `p >= 1`.
    pub fn logseries(&mut self, p: f64, shape: &[usize]) -> Array<i64> {
        assert!((0.0..1.0).contains(&p), "p < 0, p >= 1 or p is NaN");
        let len: usize = shape.iter().product();
        Array::from_vec(engine::logseries_vec(&mut self.inner, p, len), shape)
    }

    /// Zipf distribution, like `rng.zipf(a, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let arr = rng.zipf(2.5, &[1000]);
    /// assert_eq!(arr.shape(), &[1000]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `a <= 1`.
    pub fn zipf(&mut self, a: f64, shape: &[usize]) -> Array<i64> {
        assert!(a > 1.0, "a <= 1 or a is NaN");
        let len: usize = shape.iter().product();
        Array::from_vec(engine::zipf_vec(&mut self.inner, a, len), shape)
    }

    /// Multinomial distribution, like `rng.multinomial(n, pvals, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.multinomial(10, &[0.2, 0.3, 0.5], &[5]);
    /// assert_eq!(a.shape(), &[5, 3]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `n < 0`, `pvals` is empty, or `pvals` has elements outside `[0, 1]`.
    pub fn multinomial(&mut self, n: i64, pvals: &[f64], shape: &[usize]) -> Array<i64> {
        assert!(n >= 0, "n < 0");
        assert!(
            !pvals.is_empty(),
            "pvals must have at least 1 dimension and the last dimension of pvals must be greater than 0."
        );
        assert!(
            pvals.iter().all(|&p| (0.0..=1.0).contains(&p)),
            "pvals < 0, pvals > 1 or pvals contains NaNs"
        );
        let k = pvals.len();
        let count: usize = shape.iter().product();
        let actual_count = if shape.is_empty() { 1 } else { count };
        let data = engine::multinomial_vec(&mut self.inner, n, pvals, actual_count);
        let out_shape = if shape.is_empty() {
            vec![k]
        } else {
            let mut s = shape.to_vec();
            s.push(k);
            s
        };
        Array::from_vec(data, &out_shape)
    }

    /// Dirichlet distribution, like `rng.dirichlet(alpha, shape)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = rng.dirichlet(&[1.0, 2.0, 3.0], &[5]);
    /// assert_eq!(a.shape(), &[5, 3]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if any element of `alpha < 0`.
    pub fn dirichlet(&mut self, alpha: &[f64], shape: &[usize]) -> Array<f64> {
        assert!(alpha.iter().all(|&a| a >= 0.0), "alpha < 0");
        let k = alpha.len();
        if k == 0 {
            return Array::from_vec(Vec::new(), &[0]);
        }
        let count: usize = shape.iter().product();
        let actual_count = if shape.is_empty() { 1 } else { count };
        let data = engine::dirichlet_vec(&mut self.inner, alpha, actual_count);
        let out_shape = if shape.is_empty() {
            vec![k]
        } else {
            let mut s = shape.to_vec();
            s.push(k);
            s
        };
        Array::from_vec(data, &out_shape)
    }

    /// Multivariate normal distribution, like `rng.multivariate_normal(mean, cov, shape, method)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let mean = [0.0, 1.0];
    /// let cov = np::array![[1.0, 0.5], [0.5, 1.0]];
    /// let a = rng.multivariate_normal(&mean, &cov, &[100], "cholesky");
    /// assert_eq!(a.shape(), &[100, 2]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `cov` is not 2-dimensional square matching `mean.len()`, or if `method` is invalid.
    pub fn multivariate_normal(
        &mut self,
        mean: &[f64],
        cov: &Array<f64>,
        shape: &[usize],
        method: &str,
    ) -> Array<f64> {
        let dim = mean.len();
        assert!(
            cov.ndim() == 2 && cov.shape()[0] == cov.shape()[1],
            "cov must be 2 dimensional and square"
        );
        assert_eq!(cov.shape()[0], dim, "mean and cov must have same length");
        assert!(
            matches!(method, "cholesky" | "eigh" | "svd"),
            "method must be one of {{'eigh', 'svd', 'cholesky'}}"
        );

        let factor: Array<f64> = match method {
            "cholesky" => crate::linalg::cholesky(cov).expect("cholesky failed"),
            "eigh" => {
                let (w, v) = crate::linalg::eigh(cov);
                let mut f_data = vec![0.0; dim * dim];
                for r in 0..dim {
                    for c in 0..dim {
                        f_data[r * dim + c] = v[[r, c]] * (w[[c]].max(0.0)).sqrt();
                    }
                }
                Array::from_vec(f_data, &[dim, dim])
            }
            "svd" => {
                let (u, s, _) = crate::linalg::svd(cov);
                let mut f_data = vec![0.0; dim * dim];
                for r in 0..dim {
                    for c in 0..dim {
                        f_data[r * dim + c] = u[[r, c]] * (s[[c]].max(0.0)).sqrt();
                    }
                }
                Array::from_vec(f_data, &[dim, dim])
            }
            _ => unreachable!(),
        };

        let count: usize = shape.iter().product();
        let actual_count = if shape.is_empty() { 1 } else { count };
        let total_normals = actual_count * dim;
        let normals = engine::standard_normal_vec(&mut self.inner, total_normals);
        let mut out_data = Vec::with_capacity(total_normals);
        for s in 0..actual_count {
            let z = &normals[s * dim..(s + 1) * dim];
            for i in 0..dim {
                let mut val = mean[i];
                for j in 0..dim {
                    val += factor[[i, j]] * z[j];
                }
                out_data.push(val);
            }
        }
        let out_shape = if shape.is_empty() {
            vec![dim]
        } else {
            let mut s = shape.to_vec();
            s.push(dim);
            s
        };
        Array::from_vec(out_data, &out_shape)
    }

    /// Random bytes, like `rng.bytes(length)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let b = rng.bytes(10);
    /// assert_eq!(b.len(), 10);
    /// ```
    pub fn bytes(&mut self, length: usize) -> Vec<u8> {
        engine::bytes_vec(&mut self.inner, length)
    }

    /// Permuted copy of array along axis, like `rng.permuted(x, axis)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustnum as np;
    /// let mut rng = np::random::Rng::seed(42);
    /// let a = np::arange(0.0, 6.0, 1.0).reshape(&[2, 3]);
    /// let p = rng.permuted(&a, Some(1));
    /// assert_eq!(p.shape(), &[2, 3]);
    /// ```
    pub fn permuted<T: Element>(&mut self, x: &Array<T>, axis: Option<isize>) -> Array<T> {
        let ndim = x.ndim();
        match axis {
            None => {
                let mut data = x.to_vec();
                engine::shuffle(&mut self.inner, &mut data);
                Array::from_vec(data, x.shape())
            }
            Some(raw_ax) => {
                assert!(
                    ndim > 0,
                    "permuted with axis requires at least 1-dimensional array"
                );
                assert!(
                    (0..ndim as isize).contains(&raw_ax) || (-(ndim as isize)..0).contains(&raw_ax),
                    "axis {raw_ax} is out of bounds for array of dimension {ndim}"
                );
                let ax = if raw_ax < 0 {
                    (ndim as isize + raw_ax) as usize
                } else {
                    raw_ax as usize
                };
                let shape = x.shape();
                let axis_len = shape[ax];
                let total_len: usize = shape.iter().product();
                if total_len == 0 || axis_len <= 1 {
                    return x.clone();
                }
                let mut stride = 1usize;
                for &dim in &shape[ax + 1..] {
                    stride *= dim;
                }
                let mut out_data = x.to_vec();
                let num_slices = total_len / axis_len;
                for s in 0..num_slices {
                    let high = s / stride;
                    let low = s % stride;
                    let base = high * (axis_len * stride) + low;
                    let perm = engine::permutation_vec(&mut self.inner, axis_len);
                    let mut orig_slice = Vec::with_capacity(axis_len);
                    for j in 0..axis_len {
                        orig_slice.push(out_data[base + j * stride]);
                    }
                    for j in 0..axis_len {
                        out_data[base + j * stride] = orig_slice[perm[j]];
                    }
                }
                Array::from_vec(out_data, shape)
            }
        }
    }
}
