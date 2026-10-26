//! Measures the max ulp error of every SIMD kernel in `simd_math` (AVX2 and scalar paths) against `f64` std.
#[path = "../src/simd_math.rs"]
mod simd_math;

fn ulp64(x: f64, r: f64) -> f64 {
    if x == r || (x.is_nan() && r.is_nan()) {
        return 0.0;
    }
    if !x.is_finite() || !r.is_finite() {
        return f64::INFINITY;
    }
    let spacing = if r == 0.0 {
        f64::from_bits(1)
    } else {
        let e = r.abs().log2().floor();
        2f64.powf(e - 52.0).max(f64::from_bits(1))
    };
    (x - r).abs() / spacing
}
fn ulp32(x: f32, r: f64) -> f64 {
    let rr = r as f32;
    if x == rr || (x.is_nan() && rr.is_nan()) {
        return 0.0;
    }
    if !x.is_finite() || !rr.is_finite() {
        return f64::INFINITY;
    }
    let spacing = if r == 0.0 {
        f32::from_bits(1) as f64
    } else {
        let e = r.abs().log2().floor();
        2f64.powf(e - 23.0).max(f32::from_bits(1) as f64)
    };
    ((x as f64) - r).abs() / spacing
}
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn main() {
    const N: usize = 2_000_000;
    type F64 = fn(&[f64], &mut [f64]);
    type F32 = fn(&[f32], &mut [f32]);
    type Case = (&'static str, F64, F32, fn(f64) -> f64, f64, f64);
    let cases: Vec<Case> = vec![
        (
            "exp",
            simd_math::exp_f64,
            simd_math::exp_f32,
            f64::exp,
            -740.0,
            709.0,
        ),
        (
            "exp2",
            simd_math::exp2_f64,
            simd_math::exp2_f32,
            f64::exp2,
            -1070.0,
            1023.0,
        ),
        (
            "expm1",
            simd_math::expm1_f64,
            simd_math::expm1_f32,
            f64::exp_m1,
            -40.0,
            709.0,
        ),
        (
            "log",
            simd_math::log_f64,
            simd_math::log_f32,
            f64::ln,
            1e-300,
            1e300,
        ),
        (
            "log2",
            simd_math::log2_f64,
            simd_math::log2_f32,
            f64::log2,
            1e-300,
            1e300,
        ),
        (
            "log10",
            simd_math::log10_f64,
            simd_math::log10_f32,
            f64::log10,
            1e-300,
            1e300,
        ),
        (
            "log1p",
            simd_math::log1p_f64,
            simd_math::log1p_f32,
            f64::ln_1p,
            -0.999,
            1e300,
        ),
        (
            "sin",
            simd_math::sin_f64,
            simd_math::sin_f32,
            f64::sin,
            -1e4,
            1e4,
        ),
        (
            "cos",
            simd_math::cos_f64,
            simd_math::cos_f32,
            f64::cos,
            -1e4,
            1e4,
        ),
        (
            "tan",
            simd_math::tan_f64,
            simd_math::tan_f32,
            f64::tan,
            -1e4,
            1e4,
        ),
        (
            "tanh",
            simd_math::tanh_f64,
            simd_math::tanh_f32,
            f64::tanh,
            -30.0,
            30.0,
        ),
        (
            "sigmoid",
            simd_math::sigmoid_f64,
            simd_math::sigmoid_f32,
            |x| 1.0 / (1.0 + (-x).exp()),
            -700.0,
            700.0,
        ),
    ];
    for force_scalar in [false, true] {
        simd_math::set_force_scalar(force_scalar);
        println!(
            "== {}",
            if force_scalar {
                "scalar path"
            } else {
                "AVX2 path"
            }
        );
        for (name, f64k, f32k, reference, lo, hi) in &cases {
            let mut rng = Lcg(7);
            let log_scale = *lo > 0.0 && hi / lo > 1e6;
            let xs: Vec<f64> = (0..N)
                .map(|_| {
                    let u = rng.next();
                    if log_scale {
                        (lo.ln() + u * (hi.ln() - lo.ln())).exp()
                    } else {
                        lo + u * (hi - lo)
                    }
                })
                .collect();
            let mut out = vec![0.0; N];
            f64k(&xs, &mut out);
            let (mut worst, mut at) = (0.0f64, 0.0f64);
            for (&x, &y) in xs.iter().zip(&out) {
                let u = ulp64(y, reference(x));
                if u > worst {
                    worst = u;
                    at = x;
                }
            }
            let xs32: Vec<f32> = xs
                .iter()
                .map(|&x| x as f32)
                .filter(|x| x.is_finite())
                .collect();
            let mut out32 = vec![0.0f32; xs32.len()];
            f32k(&xs32, &mut out32);
            let (mut worst32, mut at32) = (0.0f64, 0.0f32);
            for (&x, &y) in xs32.iter().zip(&out32) {
                let r = reference(x as f64);
                if (r as f32).is_finite() && r.abs() > 1e-37 {
                    let u = ulp32(y, r);
                    if u > worst32 {
                        worst32 = u;
                        at32 = x;
                    }
                }
            }
            println!(
                "{name:8} f64 max {worst:>10.2} ulp (x={at:.6e})   f32 max {worst32:>10.2} ulp (x={at32:.6e})"
            );
        }
    }
}
