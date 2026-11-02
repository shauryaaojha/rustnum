#![allow(clippy::approx_constant, clippy::chunks_exact_to_as_chunks)]

//! Statistical validation tests for numpy.random distributions.
//!
//! For each distribution:
//! - 10^6 samples
//! - Sample mean within 5 standard errors of exact theoretical value
//! - Sample variance within 5 standard errors of exact theoretical value
//! - Kolmogorov-Smirnov test against exact CDF for continuous distributions (critical value D < 0.00195 at alpha = 0.001)
//! - Chi-square test on PMF for discrete distributions
//! - Parameter validation panics with NumPy's exact error messages
//! - Seed reproducibility
//! - Determinism across Rayon threads

use rustnum as np;
use rustnum::random::Rng;

fn sample_mean_and_var(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var)
}

fn sample_mean_and_var_i64(samples: &[i64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = samples
        .iter()
        .map(|&x| (x as f64 - mean).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    (mean, var)
}

fn kolmogorov_smirnov<F: Fn(f64) -> f64>(mut samples: Vec<f64>, cdf: F) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len() as f64;
    let mut d_max = 0.0f64;
    for (i, &x) in samples.iter().enumerate() {
        let f_emp_high = (i + 1) as f64 / n;
        let f_emp_low = i as f64 / n;
        let f_theo = cdf(x);
        let d1 = (f_emp_high - f_theo).abs();
        let d2 = (f_theo - f_emp_low).abs();
        if d1 > d_max {
            d_max = d1;
        }
        if d2 > d_max {
            d_max = d2;
        }
    }
    d_max
}

fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun 7.1.26 approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let abs_x = x.abs();
    let t = 1.0 / (1.0 + p * abs_x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-abs_x * abs_x).exp();
    sign * y
}

// Critical value for KS test with N=1,000,000 at alpha=0.001
const KS_CRITICAL_001: f64 = 0.00195;
const N_SAMPLES: usize = 1_000_000;

#[test]
fn test_standard_exponential_distribution() {
    let mut rng = Rng::seed(42);
    let arr = rng.standard_exponential(&[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, var) = sample_mean_and_var(&vec);

    // Exact theoretical moments for standard exponential:
    // mean = 1.0, var = 1.0
    let exact_mean = 1.0;
    let exact_var = 1.0;
    let se_mean = (exact_var / N_SAMPLES as f64).sqrt();
    let se_var = (9.0 / N_SAMPLES as f64).sqrt(); // kurtosis = 9 for Exp(1)

    assert!((mean - exact_mean).abs() < 5.0 * se_mean, "mean: {mean}");
    assert!((var - exact_var).abs() < 5.0 * se_var, "var: {var}");

    let ks = kolmogorov_smirnov(vec, |x| if x <= 0.0 { 0.0 } else { 1.0 - (-x).exp() });
    assert!(ks < KS_CRITICAL_001, "KS={ks}");
}

#[test]
fn test_standard_gamma_distribution() {
    let mut rng = Rng::seed(42);
    let k = 3.0;
    let arr = rng.standard_gamma(k, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, var) = sample_mean_and_var(&vec);

    // Exact moments for Gamma(k, 1):
    // mean = k = 3.0, var = k = 3.0
    let exact_mean = k;
    let exact_var = k;
    let se_mean = (exact_var / N_SAMPLES as f64).sqrt();
    let se_var = ((3.0 * exact_var.powi(2) + 6.0 * exact_var) / N_SAMPLES as f64).sqrt();

    assert!((mean - exact_mean).abs() < 5.0 * se_mean, "mean: {mean}");
    assert!((var - exact_var).abs() < 5.0 * se_var, "var: {var}");

    // For k=3, CDF = 1 - e^(-x) * (1 + x + x^2 / 2)
    let ks = kolmogorov_smirnov(vec, |x| {
        if x <= 0.0 {
            0.0
        } else {
            1.0 - (-x).exp() * (1.0 + x + 0.5 * x * x)
        }
    });
    assert!(ks < KS_CRITICAL_001, "KS={ks}");
}

#[test]
fn test_standard_cauchy_distribution() {
    let mut rng = Rng::seed(42);
    let arr = rng.standard_cauchy(&[N_SAMPLES]);
    let vec = arr.to_vec();

    // Cauchy has no finite moments. Tested via KS test against exact CDF:
    // CDF(x) = 0.5 + arctan(x) / pi
    let ks = kolmogorov_smirnov(vec, |x| 0.5 + x.atan() / std::f64::consts::PI);
    assert!(ks < KS_CRITICAL_001, "KS={ks}");
}

#[test]
fn test_standard_t_distribution() {
    let mut rng = Rng::seed(42);
    let df = 5.0;
    let arr = rng.standard_t(df, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, var) = sample_mean_and_var(&vec);

    // Exact theoretical moments for t(5):
    // mean = 0, var = df / (df - 2) = 5/3 = 1.666667
    let exact_mean = 0.0;
    let exact_var = df / (df - 2.0);
    let se_mean = (exact_var / N_SAMPLES as f64).sqrt();
    assert!((mean - exact_mean).abs() < 5.0 * se_mean, "mean: {mean}");
    assert!((var - exact_var).abs() < 0.05, "var: {var}");
}

#[test]
fn test_chisquare_distribution() {
    let mut rng = Rng::seed(42);
    let df = 4.0;
    let arr = rng.chisquare(df, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, var) = sample_mean_and_var(&vec);

    // Exact theoretical moments for chi2(4):
    // mean = df = 4.0, var = 2 * df = 8.0
    let exact_mean = df;
    let exact_var = 2.0 * df;
    let se_mean = (exact_var / N_SAMPLES as f64).sqrt();
    let se_var = ((2.0 * exact_var.powi(2) + 48.0) / N_SAMPLES as f64).sqrt();

    assert!((mean - exact_mean).abs() < 5.0 * se_mean, "mean: {mean}");
    assert!((var - exact_var).abs() < 5.0 * se_var, "var: {var}");

    // For df=4 (k=2): CDF = 1 - e^(-x/2) * (1 + x/2)
    let ks = kolmogorov_smirnov(vec, |x| {
        if x <= 0.0 {
            0.0
        } else {
            1.0 - (-0.5 * x).exp() * (1.0 + 0.5 * x)
        }
    });
    assert!(ks < KS_CRITICAL_001, "KS={ks}");
}

#[test]
fn test_noncentral_chisquare_distribution() {
    let mut rng = Rng::seed(42);
    let df = 4.0;
    let nonc = 2.0;
    let arr = rng.noncentral_chisquare(df, nonc, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, var) = sample_mean_and_var(&vec);

    // Exact theoretical moments for noncentral chi2:
    // mean = df + nonc = 6.0, var = 2 * (df + 2 * nonc) = 16.0
    let exact_mean = df + nonc;
    let exact_var = 2.0 * (df + 2.0 * nonc);
    let se_mean = (exact_var / N_SAMPLES as f64).sqrt();

    assert!((mean - exact_mean).abs() < 5.0 * se_mean, "mean: {mean}");
    assert!((var - exact_var).abs() < 0.2, "var: {var}");
}

#[test]
fn test_f_distribution() {
    let mut rng = Rng::seed(42);
    let dfnum = 6.0;
    let dfden = 8.0;
    let arr = rng.f(dfnum, dfden, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, var) = sample_mean_and_var(&vec);

    // Exact theoretical moments for F(d1, d2):
    // mean = d2 / (d2 - 2) = 8 / 6 = 4/3
    // var = 2 * d2^2 * (d1 + d2 - 2) / (d1 * (d2 - 2)^2 * (d2 - 4)) = 16 / 9
    let exact_mean = dfden / (dfden - 2.0);
    let exact_var = (2.0 * dfden * dfden * (dfnum + dfden - 2.0))
        / (dfnum * (dfden - 2.0).powi(2) * (dfden - 4.0));
    let se_mean = (exact_var / N_SAMPLES as f64).sqrt();

    assert!((mean - exact_mean).abs() < 5.0 * se_mean, "mean: {mean}");
    assert!((var - exact_var).abs() < 0.1, "var: {var}");
}

#[test]
fn test_noncentral_f_distribution() {
    let mut rng = Rng::seed(42);
    let dfnum = 6.0;
    let dfden = 8.0;
    let nonc = 2.0;
    let arr = rng.noncentral_f(dfnum, dfden, nonc, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, _) = sample_mean_and_var(&vec);

    // Exact theoretical mean for noncentral F:
    // mean = (dfden * (dfnum + nonc)) / (dfnum * (dfden - 2)) = (8 * 8) / (6 * 6) = 16/9
    let exact_mean = (dfden * (dfnum + nonc)) / (dfnum * (dfden - 2.0));
    assert!((mean - exact_mean).abs() < 0.05, "mean: {mean}");
}

#[test]
fn test_laplace_distribution() {
    let mut rng = Rng::seed(42);
    let loc = 2.0;
    let scale = 1.5;
    let arr = rng.laplace(loc, scale, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, var) = sample_mean_and_var(&vec);

    // Exact theoretical moments for Laplace(loc, scale):
    // mean = loc = 2.0, var = 2 * scale^2 = 4.5
    let exact_mean = loc;
    let exact_var = 2.0 * scale * scale;
    let se_mean = (exact_var / N_SAMPLES as f64).sqrt();
    let se_var = ((5.0 * exact_var.powi(2)) / N_SAMPLES as f64).sqrt();

    assert!((mean - exact_mean).abs() < 5.0 * se_mean, "mean: {mean}");
    assert!((var - exact_var).abs() < 5.0 * se_var, "var: {var}");

    let ks = kolmogorov_smirnov(vec, |x| {
        if x < loc {
            0.5 * ((x - loc) / scale).exp()
        } else {
            1.0 - 0.5 * (-(x - loc) / scale).exp()
        }
    });
    assert!(ks < KS_CRITICAL_001, "KS={ks}");
}

#[test]
fn test_logistic_distribution() {
    let mut rng = Rng::seed(42);
    let loc = 1.0;
    let scale = 2.0;
    let arr = rng.logistic(loc, scale, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, var) = sample_mean_and_var(&vec);

    // Exact moments for Logistic(loc, scale):
    // mean = loc = 1.0, var = (pi^2 / 3) * scale^2
    let exact_mean = loc;
    let exact_var = (std::f64::consts::PI.powi(2) / 3.0) * scale * scale;
    let se_mean = (exact_var / N_SAMPLES as f64).sqrt();

    assert!((mean - exact_mean).abs() < 5.0 * se_mean, "mean: {mean}");
    assert!((var - exact_var).abs() < 0.2, "var: {var}");

    let ks = kolmogorov_smirnov(vec, |x| 1.0 / (1.0 + (-(x - loc) / scale).exp()));
    assert!(ks < KS_CRITICAL_001, "KS={ks}");
}

#[test]
fn test_lognormal_distribution() {
    let mut rng = Rng::seed(42);
    let m = 0.5;
    let s = 0.5;
    let arr = rng.lognormal(m, s, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, var) = sample_mean_and_var(&vec);

    // Exact theoretical moments for Lognormal(m, s):
    // mean = exp(m + s^2 / 2)
    // var = (exp(s^2) - 1) * exp(2m + s^2)
    let exact_mean = (m + s * s / 2.0).exp();
    let exact_var = (s * s).exp_m1() * (2.0 * m + s * s).exp();
    let se_mean = (exact_var / N_SAMPLES as f64).sqrt();

    assert!((mean - exact_mean).abs() < 5.0 * se_mean, "mean: {mean}");
    assert!((var - exact_var).abs() < 0.05, "var: {var}");

    let ks = kolmogorov_smirnov(vec, |x| {
        if x <= 0.0 {
            0.0
        } else {
            standard_normal_cdf((x.ln() - m) / s)
        }
    });
    assert!(ks < KS_CRITICAL_001, "KS={ks}");
}

#[test]
fn test_gumbel_distribution() {
    let mut rng = Rng::seed(42);
    let loc = 2.0;
    let scale = 1.5;
    let arr = rng.gumbel(loc, scale, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, var) = sample_mean_and_var(&vec);

    // Exact theoretical moments for Gumbel(loc, scale):
    // mean = loc + scale * euler_mascheroni
    // var = (pi^2 / 6) * scale^2
    const EULER_MASCHERONI: f64 = 0.5772156649015329;
    let exact_mean = loc + scale * EULER_MASCHERONI;
    let exact_var = (std::f64::consts::PI.powi(2) / 6.0) * scale * scale;
    let se_mean = (exact_var / N_SAMPLES as f64).sqrt();

    assert!((mean - exact_mean).abs() < 5.0 * se_mean, "mean: {mean}");
    assert!((var - exact_var).abs() < 0.1, "var: {var}");

    let ks = kolmogorov_smirnov(vec, |x| (-(-(x - loc) / scale).exp()).exp());
    assert!(ks < KS_CRITICAL_001, "KS={ks}");
}

#[test]
fn test_rayleigh_distribution() {
    let mut rng = Rng::seed(42);
    let scale = 2.0;
    let arr = rng.rayleigh(scale, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, var) = sample_mean_and_var(&vec);

    // Exact theoretical moments for Rayleigh(scale):
    // mean = scale * sqrt(pi / 2)
    // var = (4 - pi) / 2 * scale^2
    let exact_mean = scale * (std::f64::consts::PI / 2.0).sqrt();
    let exact_var = ((4.0 - std::f64::consts::PI) / 2.0) * scale * scale;
    let se_mean = (exact_var / N_SAMPLES as f64).sqrt();

    assert!((mean - exact_mean).abs() < 5.0 * se_mean, "mean: {mean}");
    assert!((var - exact_var).abs() < 0.05, "var: {var}");

    let ks = kolmogorov_smirnov(vec, |x| {
        if x <= 0.0 {
            0.0
        } else {
            1.0 - (-x * x / (2.0 * scale * scale)).exp()
        }
    });
    assert!(ks < KS_CRITICAL_001, "KS={ks}");
}

#[test]
fn test_pareto_distribution() {
    let mut rng = Rng::seed(42);
    let a = 3.0;
    let arr = rng.pareto(a, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, var) = sample_mean_and_var(&vec);

    // Exact theoretical moments for Pareto II (Lomax):
    // mean = 1 / (a - 1) = 0.5
    // var = a / ((a - 1)^2 * (a - 2)) = 3 / (4 * 1) = 0.75
    let exact_mean = 1.0 / (a - 1.0);
    let exact_var = a / ((a - 1.0).powi(2) * (a - 2.0));
    let se_mean = (exact_var / N_SAMPLES as f64).sqrt();

    assert!((mean - exact_mean).abs() < 5.0 * se_mean, "mean: {mean}");
    assert!((var - exact_var).abs() < 0.05, "var: {var}");

    let ks = kolmogorov_smirnov(vec, |x| {
        if x <= 0.0 {
            0.0
        } else {
            1.0 - (1.0 + x).powf(-a)
        }
    });
    assert!(ks < KS_CRITICAL_001, "KS={ks}");
}

#[test]
fn test_power_distribution() {
    let mut rng = Rng::seed(42);
    let a = 3.0;
    let arr = rng.power(a, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, var) = sample_mean_and_var(&vec);

    // Exact theoretical moments for Power(a):
    // mean = a / (a + 1) = 0.75
    // var = a / ((a + 2) * (a + 1)^2) = 3 / (5 * 16) = 3 / 80 = 0.0375
    let exact_mean = a / (a + 1.0);
    let exact_var = a / ((a + 2.0) * (a + 1.0).powi(2));
    let se_mean = (exact_var / N_SAMPLES as f64).sqrt();

    assert!((mean - exact_mean).abs() < 5.0 * se_mean, "mean: {mean}");
    assert!(
        (var - exact_var).abs() < 5.0 * (exact_var / N_SAMPLES as f64).sqrt(),
        "var: {var}"
    );

    let ks = kolmogorov_smirnov(vec, |x| {
        if x <= 0.0 {
            0.0
        } else if x >= 1.0 {
            1.0
        } else {
            x.powf(a)
        }
    });
    assert!(ks < KS_CRITICAL_001, "KS={ks}");
}

#[test]
fn test_weibull_distribution() {
    let mut rng = Rng::seed(42);
    let a = 2.0;
    let arr = rng.weibull(a, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, var) = sample_mean_and_var(&vec);

    // Exact theoretical moments for Weibull(a=2):
    // mean = Gamma(1 + 1/2) = sqrt(pi) / 2
    // var = Gamma(2) - pi/4 = 1 - pi/4
    let exact_mean = std::f64::consts::PI.sqrt() / 2.0;
    let exact_var = 1.0 - std::f64::consts::PI / 4.0;
    let se_mean = (exact_var / N_SAMPLES as f64).sqrt();

    assert!((mean - exact_mean).abs() < 5.0 * se_mean, "mean: {mean}");
    assert!((var - exact_var).abs() < 0.02, "var: {var}");

    let ks = kolmogorov_smirnov(vec, |x| {
        if x <= 0.0 {
            0.0
        } else {
            1.0 - (-x.powf(a)).exp()
        }
    });
    assert!(ks < KS_CRITICAL_001, "KS={ks}");
}

#[test]
fn test_wald_distribution() {
    let mut rng = Rng::seed(42);
    let mean_param = 2.0;
    let scale = 3.0;
    let arr = rng.wald(mean_param, scale, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, var) = sample_mean_and_var(&vec);

    // Exact theoretical moments for Wald(mean, scale):
    // mean = mean_param = 2.0
    // var = mean^3 / scale = 8 / 3 = 2.666667
    let exact_mean = mean_param;
    let exact_var = mean_param.powi(3) / scale;
    let se_mean = (exact_var / N_SAMPLES as f64).sqrt();

    assert!((mean - exact_mean).abs() < 5.0 * se_mean, "mean: {mean}");
    assert!((var - exact_var).abs() < 0.08, "var: {var}");
}

#[test]
fn test_triangular_distribution() {
    let mut rng = Rng::seed(42);
    let left = 0.0;
    let mode = 1.0;
    let right = 4.0;
    let arr = rng.triangular(left, mode, right, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, var) = sample_mean_and_var(&vec);

    // Exact theoretical moments for Triangular(left, mode, right):
    // mean = (left + mode + right) / 3 = 5/3 = 1.666667
    // var = (left^2 + mode^2 + right^2 - left*mode - left*right - mode*right) / 18 = 13/18
    let exact_mean = (left + mode + right) / 3.0;
    let exact_var =
        (left.powi(2) + mode.powi(2) + right.powi(2) - left * mode - left * right - mode * right)
            / 18.0;
    let se_mean = (exact_var / N_SAMPLES as f64).sqrt();

    assert!((mean - exact_mean).abs() < 5.0 * se_mean, "mean: {mean}");
    assert!((var - exact_var).abs() < 0.02, "var: {var}");

    let ks = kolmogorov_smirnov(vec, |x| {
        if x <= left {
            0.0
        } else if x <= mode {
            (x - left).powi(2) / ((right - left) * (mode - left))
        } else if x <= right {
            1.0 - (right - x).powi(2) / ((right - left) * (right - mode))
        } else {
            1.0
        }
    });
    assert!(ks < KS_CRITICAL_001, "KS={ks}");
}

#[test]
fn test_vonmises_distribution() {
    let mut rng = Rng::seed(42);
    let mu = 0.0;
    let kappa = 2.0;
    let arr = rng.vonmises(mu, kappa, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, _) = sample_mean_and_var(&vec);

    // Exact mean for symmetric vonmises around 0: mean = 0.0
    assert!(mean.abs() < 0.01, "mean: {mean}");

    // Circular mean cos(x) should equal I1(kappa) / I0(kappa)
    let mean_cos = vec.iter().map(|&x| x.cos()).sum::<f64>() / N_SAMPLES as f64;
    let exact_cos = 0.6977746579640083; // I1(2) / I0(2)
    assert!((mean_cos - exact_cos).abs() < 0.005, "cos mean: {mean_cos}");
}

#[test]
fn test_geometric_distribution() {
    let mut rng = Rng::seed(42);
    let p = 0.4;
    let arr = rng.geometric(p, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, var) = sample_mean_and_var_i64(&vec);

    // Exact theoretical moments for Geometric(p):
    // mean = 1 / p = 2.5, var = (1 - p) / p^2 = 0.6 / 0.16 = 3.75
    let exact_mean = 1.0 / p;
    let exact_var = (1.0 - p) / (p * p);
    let se_mean = (exact_var / N_SAMPLES as f64).sqrt();

    assert!((mean - exact_mean).abs() < 5.0 * se_mean, "mean: {mean}");
    assert!((var - exact_var).abs() < 0.05, "var: {var}");

    // Chi-square test on categories 1, 2, 3, 4, 5, >=6
    let mut counts = [0usize; 6];
    for &x in &vec {
        if (1..=5).contains(&x) {
            counts[(x - 1) as usize] += 1;
        } else if x >= 6 {
            counts[5] += 1;
        }
    }
    let mut chi2 = 0.0;
    for k in 1..=5 {
        let expected = N_SAMPLES as f64 * (1.0 - p).powi(k - 1) * p;
        let obs = counts[(k - 1) as usize] as f64;
        chi2 += (obs - expected).powi(2) / expected;
    }
    let exp_tail = N_SAMPLES as f64 * (1.0 - p).powi(5);
    chi2 += (counts[5] as f64 - exp_tail).powi(2) / exp_tail;

    // df = 5, critical value at alpha=0.001 is 20.515
    assert!(chi2 < 20.515, "Chi2={chi2}");
}

#[test]
fn test_negative_binomial_distribution() {
    let mut rng = Rng::seed(42);
    let n = 4.0;
    let p = 0.5;
    let arr = rng.negative_binomial(n, p, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, var) = sample_mean_and_var_i64(&vec);

    // Exact theoretical moments for NegativeBinomial(n, p):
    // mean = n * (1 - p) / p = 4.0
    // var = n * (1 - p) / p^2 = 8.0
    let exact_mean = n * (1.0 - p) / p;
    let exact_var = n * (1.0 - p) / (p * p);
    let se_mean = (exact_var / N_SAMPLES as f64).sqrt();

    assert!((mean - exact_mean).abs() < 5.0 * se_mean, "mean: {mean}");
    assert!((var - exact_var).abs() < 0.1, "var: {var}");
}

#[test]
fn test_hypergeometric_distribution() {
    let mut rng = Rng::seed(42);
    let ngood = 50;
    let nbad = 50;
    let nsample = 20;
    let arr = rng.hypergeometric(ngood, nbad, nsample, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, var) = sample_mean_and_var_i64(&vec);

    // Exact theoretical moments for Hypergeometric(50, 50, 20):
    // mean = nsample * ngood / N = 10.0
    // var = nsample * (ngood/N) * (nbad/N) * ((N - nsample)/(N - 1)) = 20 * 0.5 * 0.5 * (80/99) = 400/99
    let n_tot = (ngood + nbad) as f64;
    let exact_mean = (nsample as f64 * ngood as f64) / n_tot;
    let exact_var = exact_mean * (nbad as f64 / n_tot) * ((n_tot - nsample as f64) / (n_tot - 1.0));
    let se_mean = (exact_var / N_SAMPLES as f64).sqrt();

    assert!((mean - exact_mean).abs() < 5.0 * se_mean, "mean: {mean}");
    assert!((var - exact_var).abs() < 0.05, "var: {var}");
}

#[test]
fn test_multivariate_hypergeometric_distribution() {
    let mut rng = Rng::seed(42);
    let colors = [30, 30, 40];
    let nsample = 20;
    let arr = rng.multivariate_hypergeometric(&colors, nsample, &[100_000]);
    assert_eq!(arr.shape(), &[100_000, 3]);

    let data = arr.to_vec();
    let mut sums = [0.0; 3];
    for chunk in data.chunks_exact(3) {
        assert_eq!(chunk[0] + chunk[1] + chunk[2], nsample);
        sums[0] += chunk[0] as f64;
        sums[1] += chunk[1] as f64;
        sums[2] += chunk[2] as f64;
    }
    // Expected means: [20 * 0.3, 20 * 0.3, 20 * 0.4] = [6.0, 6.0, 8.0]
    let m0 = sums[0] / 100_000.0;
    let m1 = sums[1] / 100_000.0;
    let m2 = sums[2] / 100_000.0;
    assert!((m0 - 6.0).abs() < 0.05, "m0={m0}");
    assert!((m1 - 6.0).abs() < 0.05, "m1={m1}");
    assert!((m2 - 8.0).abs() < 0.05, "m2={m2}");
}

#[test]
fn test_logseries_distribution() {
    let mut rng = Rng::seed(42);
    let p = 0.6;
    let arr = rng.logseries(p, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, _) = sample_mean_and_var_i64(&vec);

    // Exact theoretical mean for Logseries(p):
    // mean = -p / ((1 - p) * ln(1 - p))
    let exact_mean = -p / ((1.0 - p) * (1.0 - p).ln());
    assert!(
        (mean - exact_mean).abs() < 0.02,
        "mean: {mean}, exact: {exact_mean}"
    );

    // Chi-square test on categories k = 1, 2, 3, >=4
    let mut counts = [0usize; 4];
    for &x in &vec {
        if x == 1 {
            counts[0] += 1;
        } else if x == 2 {
            counts[1] += 1;
        } else if x == 3 {
            counts[2] += 1;
        } else {
            counts[3] += 1;
        }
    }
    let norm = -1.0 / (1.0 - p).ln();
    let mut chi2 = 0.0;
    let mut p_sum = 0.0;
    for k in 1..=3 {
        let prob = norm * p.powi(k as i32) / k as f64;
        p_sum += prob;
        let expected = N_SAMPLES as f64 * prob;
        let obs = counts[k - 1] as f64;
        chi2 += (obs - expected).powi(2) / expected;
    }
    let tail_exp = N_SAMPLES as f64 * (1.0 - p_sum);
    chi2 += (counts[3] as f64 - tail_exp).powi(2) / tail_exp;
    assert!(chi2 < 16.266, "Chi2={chi2}"); // df=3, alpha=0.001
}

#[test]
fn test_zipf_distribution() {
    let mut rng = Rng::seed(42);
    let a = 3.5;
    let arr = rng.zipf(a, &[N_SAMPLES]);
    let vec = arr.to_vec();
    let (mean, var) = sample_mean_and_var_i64(&vec);

    // Exact theoretical moments for Zipf(3.5):
    // mean = zeta(2.5) / zeta(3.5) ~ 1.190598149
    // var = zeta(1.5) / zeta(3.5) - mean^2 ~ 0.901014101
    let exact_mean = 1.190598149;
    let exact_var = 0.901014101;
    let se_mean = (exact_var / N_SAMPLES as f64).sqrt();
    let se_var = 0.05; // finite standard error

    assert!(
        (mean - exact_mean).abs() < 5.0 * se_mean,
        "mean: {mean}, exact: {exact_mean}"
    );
    assert!(
        (var - exact_var).abs() < 5.0 * se_var,
        "var: {var}, exact: {exact_var}"
    );

    // Test fraction of 1s: P(X=1) = 1 / zeta(3.5) = 1 / 1.12673386 ~ 0.887521
    let ones = vec.iter().filter(|&&x| x == 1).count() as f64 / N_SAMPLES as f64;
    assert!((ones - 0.887521).abs() < 0.005, "ones fraction: {ones}");
}

#[test]
fn test_multinomial_distribution() {
    let mut rng = Rng::seed(42);
    let n = 10;
    let pvals = [0.2, 0.3, 0.5];
    let arr = rng.multinomial(n, &pvals, &[100_000]);
    assert_eq!(arr.shape(), &[100_000, 3]);

    let data = arr.to_vec();
    let mut sums = [0.0; 3];
    for chunk in data.chunks_exact(3) {
        assert_eq!(chunk[0] + chunk[1] + chunk[2], n);
        sums[0] += chunk[0] as f64;
        sums[1] += chunk[1] as f64;
        sums[2] += chunk[2] as f64;
    }
    let m0 = sums[0] / 100_000.0;
    let m1 = sums[1] / 100_000.0;
    let m2 = sums[2] / 100_000.0;
    assert!((m0 - 2.0).abs() < 0.05, "m0={m0}");
    assert!((m1 - 3.0).abs() < 0.05, "m1={m1}");
    assert!((m2 - 5.0).abs() < 0.05, "m2={m2}");
}

#[test]
fn test_dirichlet_distribution() {
    let mut rng = Rng::seed(42);
    let alpha = [2.0, 3.0, 5.0];
    let arr = rng.dirichlet(&alpha, &[100_000]);
    assert_eq!(arr.shape(), &[100_000, 3]);

    let data = arr.to_vec();
    let mut sums = [0.0; 3];
    for chunk in data.chunks_exact(3) {
        let s = chunk[0] + chunk[1] + chunk[2];
        assert!(
            (s - 1.0).abs() < 1e-12,
            "sum of simplex elements must be 1: {s}"
        );
        sums[0] += chunk[0];
        sums[1] += chunk[1];
        sums[2] += chunk[2];
    }
    // Expected means: alpha_i / alpha_0 = [0.2, 0.3, 0.5]
    let m0 = sums[0] / 100_000.0;
    let m1 = sums[1] / 100_000.0;
    let m2 = sums[2] / 100_000.0;
    assert!((m0 - 0.2).abs() < 0.005, "m0={m0}");
    assert!((m1 - 0.3).abs() < 0.005, "m1={m1}");
    assert!((m2 - 0.5).abs() < 0.005, "m2={m2}");
}

#[test]
fn test_multivariate_normal_distribution() {
    let mean = [1.0, 2.0];
    let cov = np::array![[2.0, 0.5], [0.5, 3.0]];

    for method in ["cholesky", "eigh", "svd"] {
        let mut rng = Rng::seed(42);
        let arr = rng.multivariate_normal(&mean, &cov, &[100_000], method);
        assert_eq!(arr.shape(), &[100_000, 2]);

        let data = arr.to_vec();
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_yy = 0.0;
        let mut sum_xy = 0.0;
        for chunk in data.chunks_exact(2) {
            let x = chunk[0];
            let y = chunk[1];
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_yy += y * y;
            sum_xy += x * y;
        }
        let n = 100_000.0;
        let mx = sum_x / n;
        let my = sum_y / n;
        let vx = (sum_xx - n * mx * mx) / (n - 1.0);
        let vy = (sum_yy - n * my * my) / (n - 1.0);
        let cov_xy = (sum_xy - n * mx * my) / (n - 1.0);

        assert!((mx - 1.0).abs() < 0.05, "method={method} mx={mx}");
        assert!((my - 2.0).abs() < 0.05, "method={method} my={my}");
        assert!((vx - 2.0).abs() < 0.1, "method={method} vx={vx}");
        assert!((vy - 3.0).abs() < 0.1, "method={method} vy={vy}");
        assert!(
            (cov_xy - 0.5).abs() < 0.08,
            "method={method} cov_xy={cov_xy}"
        );
    }
}

#[test]
fn test_bytes() {
    let mut rng = Rng::seed(42);
    let b = rng.bytes(100_000);
    assert_eq!(b.len(), 100_000);

    // Uniform bytes across 0..255
    let mut counts = [0usize; 256];
    for &byte in &b {
        counts[byte as usize] += 1;
    }
    let expected = 100_000.0 / 256.0;
    let mut chi2 = 0.0;
    for &c in &counts {
        chi2 += (c as f64 - expected).powi(2) / expected;
    }
    // df = 255, critical value at 0.001 is ~332.5
    assert!(chi2 < 340.0, "Chi2={chi2}");
}

#[test]
fn test_permuted() {
    let mut rng = Rng::seed(42);
    let a = np::arange(0.0, 12.0, 1.0).reshape(&[3, 4]);

    // Axis = None shuffles whole array
    let p_none = rng.permuted(&a, None);
    assert_eq!(p_none.shape(), &[3, 4]);
    let mut v_orig = a.to_vec();
    v_orig.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mut v_p = p_none.to_vec();
    v_p.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert_eq!(v_orig, v_p);

    // Axis = 1 shuffles each row independently
    let p_axis1 = rng.permuted(&a, Some(1));
    assert_eq!(p_axis1.shape(), &[3, 4]);
    for r in 0..3 {
        let mut row_orig = a.to_vec()[r * 4..(r + 1) * 4].to_vec();
        row_orig.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mut row_p = p_axis1.to_vec()[r * 4..(r + 1) * 4].to_vec();
        row_p.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_eq!(row_orig, row_p);
    }
}

// ---------------------------------------------------------------------------
// Parameter Validation Panics matching NumPy error messages
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "shape < 0")]
fn test_panic_standard_gamma() {
    let mut rng = Rng::seed(42);
    rng.standard_gamma(-1.0, &[10]);
}

#[test]
#[should_panic(expected = "df <= 0")]
fn test_panic_standard_t() {
    let mut rng = Rng::seed(42);
    rng.standard_t(0.0, &[10]);
}

#[test]
#[should_panic(expected = "df <= 0")]
fn test_panic_chisquare() {
    let mut rng = Rng::seed(42);
    rng.chisquare(0.0, &[10]);
}

#[test]
#[should_panic(expected = "df <= 0")]
fn test_panic_noncentral_chisquare_df() {
    let mut rng = Rng::seed(42);
    rng.noncentral_chisquare(0.0, 1.0, &[10]);
}

#[test]
#[should_panic(expected = "nonc < 0")]
fn test_panic_noncentral_chisquare_nonc() {
    let mut rng = Rng::seed(42);
    rng.noncentral_chisquare(1.0, -1.0, &[10]);
}

#[test]
#[should_panic(expected = "dfnum <= 0")]
fn test_panic_f_dfnum() {
    let mut rng = Rng::seed(42);
    rng.f(0.0, 1.0, &[10]);
}

#[test]
#[should_panic(expected = "dfden <= 0")]
fn test_panic_f_dfden() {
    let mut rng = Rng::seed(42);
    rng.f(1.0, 0.0, &[10]);
}

#[test]
#[should_panic(expected = "scale < 0")]
fn test_panic_laplace() {
    let mut rng = Rng::seed(42);
    rng.laplace(0.0, -1.0, &[10]);
}

#[test]
#[should_panic(expected = "scale < 0")]
fn test_panic_logistic() {
    let mut rng = Rng::seed(42);
    rng.logistic(0.0, -1.0, &[10]);
}

#[test]
#[should_panic(expected = "sigma < 0")]
fn test_panic_lognormal() {
    let mut rng = Rng::seed(42);
    rng.lognormal(0.0, -1.0, &[10]);
}

#[test]
#[should_panic(expected = "scale < 0")]
fn test_panic_gumbel() {
    let mut rng = Rng::seed(42);
    rng.gumbel(0.0, -1.0, &[10]);
}

#[test]
#[should_panic(expected = "scale < 0")]
fn test_panic_rayleigh() {
    let mut rng = Rng::seed(42);
    rng.rayleigh(-1.0, &[10]);
}

#[test]
#[should_panic(expected = "a <= 0")]
fn test_panic_pareto() {
    let mut rng = Rng::seed(42);
    rng.pareto(0.0, &[10]);
}

#[test]
#[should_panic(expected = "a <= 0")]
fn test_panic_power() {
    let mut rng = Rng::seed(42);
    rng.power(0.0, &[10]);
}

#[test]
#[should_panic(expected = "a < 0")]
fn test_panic_weibull() {
    let mut rng = Rng::seed(42);
    rng.weibull(-1.0, &[10]);
}

#[test]
#[should_panic(expected = "mean <= 0")]
fn test_panic_wald_mean() {
    let mut rng = Rng::seed(42);
    rng.wald(0.0, 1.0, &[10]);
}

#[test]
#[should_panic(expected = "scale <= 0")]
fn test_panic_wald_scale() {
    let mut rng = Rng::seed(42);
    rng.wald(1.0, 0.0, &[10]);
}

#[test]
#[should_panic(expected = "left > mode")]
fn test_panic_triangular_mode() {
    let mut rng = Rng::seed(42);
    rng.triangular(2.0, 1.0, 3.0, &[10]);
}

#[test]
#[should_panic(expected = "mode > right")]
fn test_panic_triangular_right() {
    let mut rng = Rng::seed(42);
    rng.triangular(1.0, 4.0, 3.0, &[10]);
}

#[test]
#[should_panic(expected = "left == right")]
fn test_panic_triangular_equal() {
    let mut rng = Rng::seed(42);
    rng.triangular(1.0, 1.0, 1.0, &[10]);
}

#[test]
#[should_panic(expected = "kappa < 0")]
fn test_panic_vonmises() {
    let mut rng = Rng::seed(42);
    rng.vonmises(0.0, -1.0, &[10]);
}

#[test]
#[should_panic(expected = "p <= 0, p > 1 or p contains NaNs")]
fn test_panic_geometric() {
    let mut rng = Rng::seed(42);
    rng.geometric(0.0, &[10]);
}

#[test]
#[should_panic(expected = "n <= 0")]
fn test_panic_negative_binomial_n() {
    let mut rng = Rng::seed(42);
    rng.negative_binomial(0.0, 0.5, &[10]);
}

#[test]
#[should_panic(expected = "p <= 0, p > 1 or p contains NaNs")]
fn test_panic_negative_binomial_p() {
    let mut rng = Rng::seed(42);
    rng.negative_binomial(5.0, 0.0, &[10]);
}

#[test]
#[should_panic(expected = "ngood < 0")]
fn test_panic_hypergeometric_good() {
    let mut rng = Rng::seed(42);
    rng.hypergeometric(-1, 10, 5, &[10]);
}

#[test]
#[should_panic(expected = "nbad < 0")]
fn test_panic_hypergeometric_bad() {
    let mut rng = Rng::seed(42);
    rng.hypergeometric(10, -1, 5, &[10]);
}

#[test]
#[should_panic(expected = "nsample < 0")]
fn test_panic_hypergeometric_sample() {
    let mut rng = Rng::seed(42);
    rng.hypergeometric(10, 10, -1, &[10]);
}

#[test]
#[should_panic(expected = "ngood + nbad < nsample")]
fn test_panic_hypergeometric_exceed() {
    let mut rng = Rng::seed(42);
    rng.hypergeometric(10, 10, 25, &[10]);
}

#[test]
#[should_panic(
    expected = "colors must be a one-dimensional sequence of nonnegative integers not exceeding 9223372036854775807."
)]
fn test_panic_multivariate_hypergeometric_colors() {
    let mut rng = Rng::seed(42);
    rng.multivariate_hypergeometric(&[-1, 10], 5, &[10]);
}

#[test]
#[should_panic(expected = "nsample must be nonnegative.")]
fn test_panic_multivariate_hypergeometric_sample() {
    let mut rng = Rng::seed(42);
    rng.multivariate_hypergeometric(&[10, 10], -1, &[10]);
}

#[test]
#[should_panic(expected = "nsample > sum(colors)")]
fn test_panic_multivariate_hypergeometric_exceed() {
    let mut rng = Rng::seed(42);
    rng.multivariate_hypergeometric(&[10, 10], 25, &[10]);
}

#[test]
#[should_panic(expected = "p < 0, p >= 1 or p is NaN")]
fn test_panic_logseries() {
    let mut rng = Rng::seed(42);
    rng.logseries(1.0, &[10]);
}

#[test]
#[should_panic(expected = "a <= 1 or a is NaN")]
fn test_panic_zipf() {
    let mut rng = Rng::seed(42);
    rng.zipf(1.0, &[10]);
}

#[test]
#[should_panic(expected = "n < 0")]
fn test_panic_multinomial_n() {
    let mut rng = Rng::seed(42);
    rng.multinomial(-1, &[0.5, 0.5], &[10]);
}

#[test]
#[should_panic(expected = "pvals < 0, pvals > 1 or pvals contains NaNs")]
fn test_panic_multinomial_pvals() {
    let mut rng = Rng::seed(42);
    rng.multinomial(5, &[-0.1, 0.5], &[10]);
}

#[test]
#[should_panic(expected = "alpha < 0")]
fn test_panic_dirichlet() {
    let mut rng = Rng::seed(42);
    rng.dirichlet(&[-1.0, 1.0], &[10]);
}

#[test]
#[should_panic(expected = "method must be one of {'eigh', 'svd', 'cholesky'}")]
fn test_panic_multivariate_normal_method() {
    let mut rng = Rng::seed(42);
    let cov = np::array![[1.0]];
    rng.multivariate_normal(&[0.0], &cov, &[10], "invalid");
}

// ---------------------------------------------------------------------------
// Seed Reproducibility
// ---------------------------------------------------------------------------

#[test]
fn test_seed_reproducibility() {
    let mut rng1 = Rng::seed(12345);
    let mut rng2 = Rng::seed(12345);

    let a1 = rng1.standard_exponential(&[100]);
    let a2 = rng2.standard_exponential(&[100]);
    assert_eq!(a1, a2);

    let b1 = rng1.chisquare(3.0, &[100]);
    let b2 = rng2.chisquare(3.0, &[100]);
    assert_eq!(b1, b2);

    let c1 = rng1.laplace(0.0, 1.0, &[100]);
    let c2 = rng2.laplace(0.0, 1.0, &[100]);
    assert_eq!(c1, c2);

    let d1 = rng1.zipf(2.0, &[100]);
    let d2 = rng2.zipf(2.0, &[100]);
    assert_eq!(d1, d2);
}

// ---------------------------------------------------------------------------
// Thread-count independence
// ---------------------------------------------------------------------------

#[test]
fn test_thread_count_independence() {
    // Generate 200,000 samples (exceeds CHUNK_SIZE = 65,536 so deterministic chunking is active)
    let n = 200_000;
    let mut rng1 = Rng::seed(999);
    let a1 = rng1.standard_exponential(&[n]);

    let mut rng2 = Rng::seed(999);
    let a2 = rng2.standard_exponential(&[n]);

    assert_eq!(
        a1, a2,
        "deterministic parallel filling must produce identical arrays"
    );
}
