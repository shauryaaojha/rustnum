//! Random number generation engine: xoshiro256** seeded by SplitMix64.
//!
//! All functions produce plain `Vec`s or operate on slices — no `Array`
//! dependency. The public `Rng` wrapper in `api.rs` calls these and returns `Array`.

// ---------------------------------------------------------------------------
// SplitMix64 — used only to seed xoshiro256**
// ---------------------------------------------------------------------------

pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

// ---------------------------------------------------------------------------
// xoshiro256** — the main generator
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct Xoshiro256StarStar {
    s: [u64; 4],
}

#[inline]
fn rotl(x: u64, k: u32) -> u64 {
    x.rotate_left(k)
}

impl Xoshiro256StarStar {
    /// Seed via SplitMix64 (same algorithm as the reference C implementation).
    pub(crate) fn seed(seed: u64) -> Self {
        let mut sm = SplitMix64::new(seed);
        Self {
            s: [sm.next_u64(), sm.next_u64(), sm.next_u64(), sm.next_u64()],
        }
    }

    /// Seed from system entropy: time-based + stack address.
    pub(crate) fn from_entropy() -> Self {
        use std::time::SystemTime;
        let t = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xdeadbeefcafe1234);
        // Mix in a stack address for extra entropy
        let addr = &t as *const u64 as u64;
        let seed = t ^ addr.wrapping_mul(0x517cc1b727220a95);
        Self::seed(seed)
    }

    /// Generate the next random u64.
    pub(crate) fn next_u64(&mut self) -> u64 {
        let result = rotl(self.s[1].wrapping_mul(5), 7).wrapping_mul(9);
        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = rotl(self.s[3], 45);

        result
    }

    /// Generate a random f64 in [0, 1).
    pub(crate) fn next_f64(&mut self) -> f64 {
        // Use the upper 53 bits for a uniform double in [0, 1)
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

// ---------------------------------------------------------------------------
// Distribution helpers
// ---------------------------------------------------------------------------

/// Uniform f64 in [low, high).
pub(crate) fn uniform_f64(rng: &mut Xoshiro256StarStar, low: f64, high: f64) -> f64 {
    low + (high - low) * rng.next_f64()
}

/// Unbiased bounded u64 in [0, range) using Lemire's method.
pub(crate) fn bounded_u64(rng: &mut Xoshiro256StarStar, range: u64) -> u64 {
    if range == 0 {
        return 0;
    }
    // Lemire's nearly-divisionless method
    let mut x = rng.next_u64();
    let mut m = (x as u128).wrapping_mul(range as u128);
    let mut l = m as u64;
    if l < range {
        let threshold = range.wrapping_neg() % range;
        while l < threshold {
            x = rng.next_u64();
            m = (x as u128).wrapping_mul(range as u128);
            l = m as u64;
        }
    }
    (m >> 64) as u64
}

/// Box-Muller transform: generate a pair of independent standard normals.
pub(crate) fn normal_pair(rng: &mut Xoshiro256StarStar) -> (f64, f64) {
    loop {
        let u1 = rng.next_f64();
        if u1 <= f64::MIN_POSITIVE {
            continue; // avoid log(0)
        }
        let u2 = rng.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        return (r * theta.cos(), r * theta.sin());
    }
}

/// Exponential(scale) variate.
pub(crate) fn exponential(rng: &mut Xoshiro256StarStar, scale: f64) -> f64 {
    loop {
        let u = rng.next_f64();
        if u > f64::MIN_POSITIVE {
            return -scale * u.ln();
        }
    }
}

/// Gamma(shape, scale) variate — Marsaglia & Tsang for shape >= 1,
/// Ahrens-Dieter transformation for shape < 1.
pub(crate) fn gamma(rng: &mut Xoshiro256StarStar, shape: f64, scale: f64) -> f64 {
    if shape < 1.0 {
        // Use shape+1 and multiply by U^(1/shape)
        let g = gamma(rng, shape + 1.0, 1.0);
        let u = rng.next_f64();
        return scale * g * u.powf(1.0 / shape);
    }

    // Marsaglia & Tsang's method for shape >= 1
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let (x, _) = normal_pair(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = rng.next_f64();
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return scale * d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return scale * d * v;
        }
    }
}

/// Beta(a, b) variate — via gamma.
pub(crate) fn beta(rng: &mut Xoshiro256StarStar, a: f64, b: f64) -> f64 {
    let x = gamma(rng, a, 1.0);
    let y = gamma(rng, b, 1.0);
    x / (x + y)
}

/// Binomial(n, p) by CDF inversion; only efficient (and accurate) when `n * p < 30`, `p <= 0.5`.
fn binomial_inversion(rng: &mut Xoshiro256StarStar, n: u64, p: f64) -> u64 {
    let q = 1.0 - p;
    let s = p / q;
    let a = (n + 1) as f64 * s;
    let mut r = q.powf(n as f64);
    let mut u = rng.next_f64();
    let mut x = 0u64;
    while u > r {
        u -= r;
        x += 1;
        if x > n {
            return n;
        }
        r *= a / x as f64 - s;
    }
    x
}

/// Binomial(n, p) variate, exact for every `n` and `p`.
///
/// Small `n * min(p, 1 - p)` uses inversion; larger cases split the problem with a Beta variate
/// (the order-statistic recursion from Knuth, TAOCP vol. 2) until inversion applies.
pub(crate) fn binomial(rng: &mut Xoshiro256StarStar, n: u64, p: f64) -> u64 {
    if n == 0 || p <= 0.0 {
        return 0;
    }
    if p >= 1.0 {
        return n;
    }
    if p > 0.5 {
        return n - binomial(rng, n, 1.0 - p);
    }
    let (mut n, mut p, mut acc) = (n, p, 0u64);
    while n as f64 * p >= 30.0 {
        let a = 1 + n / 2;
        let b = n + 1 - a;
        let x = beta(rng, a as f64, b as f64);
        if x >= p {
            n = a - 1;
            p /= x;
        } else {
            acc += a;
            n = b - 1;
            p = (p - x) / (1.0 - x);
        }
    }
    if n == 0 || p <= 0.0 {
        return acc;
    }
    if p > 0.5 {
        return acc + n - binomial_inversion(rng, n, 1.0 - p);
    }
    acc + binomial_inversion(rng, n, p)
}

/// Poisson(lam) variate, exact for every `lam`.
///
/// Knuth's multiplication method below 30; above that the mean is split with a Gamma variate
/// (Ahrens and Dieter) so the cost grows with `ln(lam)` rather than `lam`.
pub(crate) fn poisson(rng: &mut Xoshiro256StarStar, lam: f64) -> u64 {
    let (mut lam, mut acc) = (lam, 0u64);
    if lam.is_nan() || lam <= 0.0 {
        return 0;
    }
    while lam >= 30.0 {
        let m = (lam * 0.875).floor();
        let x = gamma(rng, m, 1.0);
        if x < lam {
            acc += m as u64;
            lam -= x;
        } else {
            return acc + binomial(rng, m as u64 - 1, lam / x);
        }
    }
    let l = (-lam).exp();
    let mut k = 0u64;
    let mut p = 1.0;
    loop {
        k += 1;
        p *= rng.next_f64();
        if p <= l {
            return acc + k - 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Vec-producing functions
// ---------------------------------------------------------------------------

/// Fill a Vec with f64 in [0, 1).
pub(crate) fn random_vec(rng: &mut Xoshiro256StarStar, len: usize) -> Vec<f64> {
    (0..len).map(|_| rng.next_f64()).collect()
}

/// Fill a Vec with uniform f64 in [low, high).
pub(crate) fn uniform_vec(
    rng: &mut Xoshiro256StarStar,
    low: f64,
    high: f64,
    len: usize,
) -> Vec<f64> {
    (0..len).map(|_| uniform_f64(rng, low, high)).collect()
}

/// Fill a Vec with normal(mean, std) f64 values.
pub(crate) fn normal_vec(
    rng: &mut Xoshiro256StarStar,
    mean: f64,
    std: f64,
    len: usize,
) -> Vec<f64> {
    let mut result = Vec::with_capacity(len);
    while result.len() < len {
        let (a, b) = normal_pair(rng);
        result.push(mean + std * a);
        if result.len() < len {
            result.push(mean + std * b);
        }
    }
    result
}

/// Fill a Vec with standard normal f64 values.
pub(crate) fn standard_normal_vec(rng: &mut Xoshiro256StarStar, len: usize) -> Vec<f64> {
    normal_vec(rng, 0.0, 1.0, len)
}

/// Fill a Vec with random i64 in [low, high).
pub(crate) fn integers_vec(
    rng: &mut Xoshiro256StarStar,
    low: i64,
    high: i64,
    len: usize,
) -> Vec<i64> {
    assert!(
        high > low,
        "high must be greater than low, got low={low}, high={high}"
    );
    let range = high.wrapping_sub(low) as u64;
    (0..len)
        .map(|_| low.wrapping_add(bounded_u64(rng, range) as i64))
        .collect()
}

/// Fill a Vec with exponential(scale) values.
pub(crate) fn exponential_vec(rng: &mut Xoshiro256StarStar, scale: f64, len: usize) -> Vec<f64> {
    (0..len).map(|_| exponential(rng, scale)).collect()
}

/// Fill a Vec with binomial(n, p) values.
pub(crate) fn binomial_vec(rng: &mut Xoshiro256StarStar, n: u64, p: f64, len: usize) -> Vec<i64> {
    (0..len).map(|_| binomial(rng, n, p) as i64).collect()
}

/// Fill a Vec with poisson(lam) values.
pub(crate) fn poisson_vec(rng: &mut Xoshiro256StarStar, lam: f64, len: usize) -> Vec<i64> {
    (0..len).map(|_| poisson(rng, lam) as i64).collect()
}

/// Fill a Vec with gamma(shape, scale) values.
pub(crate) fn gamma_vec(
    rng: &mut Xoshiro256StarStar,
    shape_param: f64,
    scale: f64,
    len: usize,
) -> Vec<f64> {
    (0..len).map(|_| gamma(rng, shape_param, scale)).collect()
}

/// Fill a Vec with beta(a, b) values.
pub(crate) fn beta_vec(rng: &mut Xoshiro256StarStar, a: f64, b: f64, len: usize) -> Vec<f64> {
    (0..len).map(|_| beta(rng, a, b)).collect()
}

/// Fisher-Yates shuffle in place.
pub(crate) fn shuffle<T>(rng: &mut Xoshiro256StarStar, slice: &mut [T]) {
    let n = slice.len();
    for i in (1..n).rev() {
        let j = bounded_u64(rng, (i + 1) as u64) as usize;
        slice.swap(i, j);
    }
}

/// Generate a permutation of 0..n.
pub(crate) fn permutation_vec(rng: &mut Xoshiro256StarStar, n: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    shuffle(rng, &mut v);
    v
}

/// Choose `k` indices from 0..n, with or without replacement.
pub(crate) fn choice_indices(
    rng: &mut Xoshiro256StarStar,
    n: usize,
    k: usize,
    replace: bool,
) -> Vec<usize> {
    if replace {
        (0..k)
            .map(|_| bounded_u64(rng, n as u64) as usize)
            .collect()
    } else {
        assert!(
            k <= n,
            "cannot take {k} samples from {n} without replacement"
        );
        if k <= n / 2 {
            // Partial Fisher-Yates
            let mut pool: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = i + bounded_u64(rng, (n - i) as u64) as usize;
                pool.swap(i, j);
            }
            pool[..k].to_vec()
        } else {
            // Full shuffle, take first k
            let mut v = permutation_vec(rng, n);
            v.truncate(k);
            v
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_determinism() {
        let mut sm = SplitMix64::new(42);
        let vals: Vec<u64> = (0..5).map(|_| sm.next_u64()).collect();
        // Verified against reference SplitMix64
        assert_eq!(
            vals,
            vec![
                13679457532755275413,
                2949826092126892291,
                5139283748462763858,
                6349198060258255764,
                701532786141963250,
            ]
        );
    }

    #[test]
    fn xoshiro256ss_determinism() {
        let mut rng = Xoshiro256StarStar::seed(42);
        let vals: Vec<u64> = (0..5).map(|_| rng.next_u64()).collect();
        // Hard-coded regression values for seed 42 checked against the
        // reference xoshiro256** algorithm seeded via SplitMix64(42).
        let expected = vec![
            1546998764402558742,
            6990951692964543102,
            12544586762248559009,
            17057574109182124193,
            18295552978065317476,
        ];
        assert_eq!(vals, expected, "seed 42 must match reference xoshiro256**");

        let mut rng2 = Xoshiro256StarStar::seed(42);
        let vals2: Vec<u64> = (0..5).map(|_| rng2.next_u64()).collect();
        assert_eq!(vals, vals2, "same seed must produce same sequence");
    }

    #[test]
    fn xoshiro256ss_from_entropy() {
        let mut rng1 = Xoshiro256StarStar::from_entropy();
        let mut rng2 = Xoshiro256StarStar::from_entropy();
        let _v1 = rng1.next_u64();
        let _v2 = rng2.next_u64();
    }

    #[test]
    fn xoshiro256ss_next_f64_range() {
        let mut rng = Xoshiro256StarStar::seed(12345);
        for _ in 0..10000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "f64 out of range: {v}");
        }
    }

    #[test]
    fn uniform_in_range() {
        let mut rng = Xoshiro256StarStar::seed(99);
        for _ in 0..10000 {
            let v = uniform_f64(&mut rng, 2.0, 5.0);
            assert!((2.0..5.0).contains(&v), "uniform out of range: {v}");
        }
    }

    #[test]
    fn normal_statistical_sanity() {
        let mut rng = Xoshiro256StarStar::seed(42);
        let n = 100_000;
        let vals = standard_normal_vec(&mut rng, n);

        let mean = vals.iter().sum::<f64>() / n as f64;
        let variance = vals.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        let std = variance.sqrt();

        assert!(mean.abs() < 0.02, "normal mean too far from 0: {mean}");
        assert!((std - 1.0).abs() < 0.02, "normal std too far from 1: {std}");
    }

    #[test]
    fn integers_in_range_and_cover() {
        let mut rng = Xoshiro256StarStar::seed(42);
        let vals = integers_vec(&mut rng, 0, 10, 10_000);

        // All in range
        for &v in &vals {
            assert!((0..10).contains(&v), "integer out of range: {v}");
        }

        // All 10 values appear
        let mut seen = [false; 10];
        for &v in &vals {
            seen[v as usize] = true;
        }
        for (i, &s) in seen.iter().enumerate() {
            assert!(s, "value {i} never appeared in 10000 samples");
        }
    }

    #[test]
    fn exponential_mean() {
        let mut rng = Xoshiro256StarStar::seed(42);
        let scale = 3.0;
        let vals = exponential_vec(&mut rng, scale, 100_000);

        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        assert!(
            (mean - scale).abs() < 0.05,
            "exponential mean too far from {scale}: {mean}"
        );

        // All positive
        for &v in &vals {
            assert!(v > 0.0, "exponential should be positive: {v}");
        }
    }

    #[test]
    fn binomial_mean() {
        let mut rng = Xoshiro256StarStar::seed(42);
        let vals = binomial_vec(&mut rng, 10, 0.3, 100_000);

        let mean = vals.iter().sum::<i64>() as f64 / vals.len() as f64;
        assert!(
            (mean - 3.0).abs() < 0.1,
            "binomial(10,0.3) mean too far from 3: {mean}"
        );
    }

    #[test]
    fn poisson_mean() {
        let mut rng = Xoshiro256StarStar::seed(42);
        let lam = 5.0;
        let vals = poisson_vec(&mut rng, lam, 100_000);

        let mean = vals.iter().sum::<i64>() as f64 / vals.len() as f64;
        assert!(
            (mean - lam).abs() < 0.1,
            "poisson(5) mean too far from 5: {mean}"
        );
    }

    fn mean_var(vals: &[i64]) -> (f64, f64) {
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<i64>() as f64 / n;
        let var = vals.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        (mean, var)
    }

    #[test]
    fn binomial_large_parameters_have_exact_moments() {
        let mut rng = Xoshiro256StarStar::seed(5);
        for &(n, p) in &[(1000u64, 0.3), (40, 0.01), (100_000, 0.9), (60, 0.5)] {
            let vals = binomial_vec(&mut rng, n, p, 40_000);
            assert!(vals.iter().all(|&v| v >= 0 && v as u64 <= n));
            let (mean, var) = mean_var(&vals);
            let (em, ev) = (n as f64 * p, n as f64 * p * (1.0 - p));
            assert!(
                (mean - em).abs() < 5.0 * (ev / 40_000.0).sqrt(),
                "n={n} p={p} mean={mean}"
            );
            assert!(
                (var / ev - 1.0).abs() < 0.06,
                "n={n} p={p} var={var} expected {ev}"
            );
        }
    }

    #[test]
    fn poisson_large_lambda_has_exact_moments() {
        let mut rng = Xoshiro256StarStar::seed(6);
        for &lam in &[29.0, 30.0, 100.0, 5000.0] {
            let vals = poisson_vec(&mut rng, lam, 40_000);
            let (mean, var) = mean_var(&vals);
            assert!(
                (mean - lam).abs() < 5.0 * (lam / 40_000.0).sqrt(),
                "lam={lam} mean={mean}"
            );
            assert!((var / lam - 1.0).abs() < 0.06, "lam={lam} var={var}");
        }
    }

    #[test]
    fn gamma_mean() {
        let mut rng = Xoshiro256StarStar::seed(42);
        let shape = 2.0;
        let scale = 3.0;
        let vals = gamma_vec(&mut rng, shape, scale, 100_000);

        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let expected = shape * scale;
        assert!(
            (mean - expected).abs() < 0.1,
            "gamma({shape},{scale}) mean too far from {expected}: {mean}"
        );
    }

    #[test]
    fn beta_mean() {
        let mut rng = Xoshiro256StarStar::seed(42);
        let a = 2.0;
        let b = 5.0;
        let vals = beta_vec(&mut rng, a, b, 100_000);

        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let expected = a / (a + b);
        assert!(
            (mean - expected).abs() < 0.02,
            "beta({a},{b}) mean too far from {expected}: {mean}"
        );

        // All in [0, 1]
        for &v in &vals {
            assert!((0.0..=1.0).contains(&v), "beta out of range: {v}");
        }
    }

    #[test]
    fn shuffle_preserves_multiset() {
        let mut rng = Xoshiro256StarStar::seed(42);
        let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut sorted_before = data.clone();
        sorted_before.sort();

        shuffle(&mut rng, &mut data);

        let mut sorted_after = data.clone();
        sorted_after.sort();
        assert_eq!(sorted_before, sorted_after);
    }

    #[test]
    fn permutation_is_permutation() {
        let mut rng = Xoshiro256StarStar::seed(42);
        let p = permutation_vec(&mut rng, 10);

        assert_eq!(p.len(), 10);
        let mut sorted = p.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn choice_without_replacement_no_duplicates() {
        let mut rng = Xoshiro256StarStar::seed(42);
        let indices = choice_indices(&mut rng, 20, 10, false);

        assert_eq!(indices.len(), 10);
        let mut sorted = indices.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            10,
            "duplicates found in choice without replacement"
        );
    }

    #[test]
    fn choice_with_replacement_length() {
        let mut rng = Xoshiro256StarStar::seed(42);
        let indices = choice_indices(&mut rng, 5, 20, true);
        assert_eq!(indices.len(), 20);
        for &i in &indices {
            assert!(i < 5);
        }
    }

    #[test]
    fn bounded_u64_range() {
        let mut rng = Xoshiro256StarStar::seed(42);
        for range in [1u64, 2, 3, 7, 100, 1000, u64::MAX] {
            for _ in 0..1000 {
                let v = bounded_u64(&mut rng, range);
                assert!(v < range, "bounded_u64({range}) produced {v}");
            }
        }
    }

    #[test]
    fn gamma_small_shape() {
        // Test gamma with shape < 1
        let mut rng = Xoshiro256StarStar::seed(42);
        let vals = gamma_vec(&mut rng, 0.5, 1.0, 10_000);
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        assert!(
            (mean - 0.5).abs() < 0.05,
            "gamma(0.5, 1) mean: {mean}, expected ~0.5"
        );
    }
}
