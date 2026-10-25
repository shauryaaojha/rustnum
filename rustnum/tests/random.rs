//! Integration tests for the public `np::random::Rng` API.

use np::random::Rng;
use rustnum as np;

fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

#[test]
fn same_seed_same_numbers_across_every_method() {
    let run = |seed| {
        let mut r = Rng::seed(seed);
        let src = np::arange(0.0, 20.0, 1.0);
        let mut shuf = np::arange(0, 10, 1);
        r.shuffle(&mut shuf);
        vec![
            format!("{:?}", r.random(&[3, 4])),
            format!("{:?}", r.uniform(-1.0, 1.0, &[5])),
            format!("{:?}", r.normal(1.0, 2.0, &[7])),
            format!("{:?}", r.standard_normal(&[2, 2])),
            format!("{:?}", r.integers(-5, 5, &[9])),
            format!("{:?}", r.choice(&src, 6, false)),
            format!("{:?}", r.choice(&src, 6, true)),
            format!("{:?}", shuf),
            format!("{:?}", r.permutation(8)),
            format!("{:?}", r.binomial(20, 0.4, &[6])),
            format!("{:?}", r.poisson(3.0, &[6])),
            format!("{:?}", r.exponential(2.0, &[6])),
            format!("{:?}", r.beta(2.0, 3.0, &[6])),
            format!("{:?}", r.gamma(2.5, 1.5, &[6])),
        ]
    };
    assert_eq!(run(42), run(42));
    assert_ne!(run(42), run(43));
}

#[test]
fn seed_zero_and_max_work_and_differ() {
    let a = Rng::seed(0).random(&[8]);
    let b = Rng::seed(u64::MAX).random(&[8]);
    assert_ne!(a, b);
    assert_eq!(a, Rng::seed(0).random(&[8]));
}

#[test]
fn successive_draws_differ_and_clone_replays_the_stream() {
    let mut r = Rng::seed(1);
    let first = r.random(&[10]);
    let mut replay = r.clone();
    let second = r.random(&[10]);
    assert_ne!(first, second);
    assert_eq!(replay.random(&[10]), second);
}

#[test]
fn splitting_a_draw_is_the_same_stream() {
    // Drawing 6 numbers at once equals drawing 2 then 4.
    let all = Rng::seed(5).random(&[6]).to_vec();
    let mut r = Rng::seed(5);
    let mut parts = r.random(&[2]).to_vec();
    parts.extend(r.random(&[4]).to_vec());
    assert_eq!(all, parts);
}

#[test]
fn from_entropy_gives_different_streams() {
    let a = Rng::from_entropy().random(&[4]);
    std::thread::sleep(std::time::Duration::from_millis(2));
    let b = Rng::from_entropy().random(&[4]);
    assert_ne!(a, b);
}

#[test]
fn shapes_and_dtypes() {
    let mut r = Rng::seed(3);
    let a: np::Array<f64> = r.random(&[2, 3, 4]);
    assert_eq!(a.shape(), &[2, 3, 4]);
    assert_eq!(a.dtype(), "float64");
    let z = r.random(&[]);
    assert_eq!(z.ndim(), 0);
    let e = r.random(&[0, 5]);
    assert_eq!((e.shape(), e.size()), (&[0, 5][..], 0));
    let i: np::Array<i64> = r.integers(0, 3, &[4, 4]);
    assert_eq!((i.shape(), i.dtype()), (&[4, 4][..], "int64"));
    let b: np::Array<i64> = r.binomial(5, 0.5, &[3]);
    let p: np::Array<i64> = r.poisson(2.0, &[3]);
    assert_eq!((b.shape(), p.shape()), (&[3][..], &[3][..]));
    let perm: np::Array<usize> = r.permutation(5);
    assert_eq!(perm.dtype(), "uint64");
    assert_eq!(r.uniform(0.0, 1.0, &[2, 2]).shape(), &[2, 2]);
    assert_eq!(r.normal(0.0, 1.0, &[3, 1]).shape(), &[3, 1]);
    assert_eq!(r.standard_normal(&[1, 3]).shape(), &[1, 3]);
    assert_eq!(r.exponential(1.0, &[2]).shape(), &[2]);
    assert_eq!(r.beta(1.0, 1.0, &[2]).shape(), &[2]);
    assert_eq!(r.gamma(1.0, 1.0, &[2]).shape(), &[2]);
}

#[test]
fn random_is_uniform_on_unit_interval() {
    let a = Rng::seed(7).random(&[100_000]).to_vec();
    assert!(a.iter().all(|v| (0.0..1.0).contains(v)));
    let (mean, std) = mean_std(&a);
    assert!((mean - 0.5).abs() < 0.01, "mean {mean}");
    assert!((std - (1.0f64 / 12.0).sqrt()).abs() < 0.01, "std {std}");
    // Each decile holds about a tenth of the draws.
    for d in 0..10 {
        let c = a.iter().filter(|&&v| (v * 10.0) as usize == d).count() as f64;
        assert!((c / 100_000.0 - 0.1).abs() < 0.006, "decile {d}: {c}");
    }
}

#[test]
fn uniform_respects_bounds_and_moments() {
    let a = Rng::seed(8).uniform(2.0, 5.0, &[50_000]).to_vec();
    assert!(a.iter().all(|&v| (2.0..5.0).contains(&v)));
    let (mean, std) = mean_std(&a);
    assert!((mean - 3.5).abs() < 0.03);
    assert!((std - 3.0 / 12f64.sqrt()).abs() < 0.03);
}

#[test]
fn normal_mean_std_sanity() {
    let a = Rng::seed(42).normal(3.0, 2.0, &[200_000]).to_vec();
    let (mean, std) = mean_std(&a);
    assert!((mean - 3.0).abs() < 0.02, "mean {mean}");
    assert!((std - 2.0).abs() < 0.02, "std {std}");
    let within_1 = a.iter().filter(|&&v| (v - 3.0).abs() < 2.0).count() as f64 / a.len() as f64;
    assert!((within_1 - 0.6827).abs() < 0.005, "68% rule: {within_1}");
    let within_2 = a.iter().filter(|&&v| (v - 3.0).abs() < 4.0).count() as f64 / a.len() as f64;
    assert!((within_2 - 0.9545).abs() < 0.004, "95% rule: {within_2}");

    let s = Rng::seed(1).standard_normal(&[100_001]).to_vec(); // odd length
    assert_eq!(s.len(), 100_001);
    let (m, sd) = mean_std(&s);
    assert!(m.abs() < 0.02 && (sd - 1.0).abs() < 0.02);
    // std = 0 is a constant array.
    assert!(
        Rng::seed(1)
            .normal(5.0, 0.0, &[10])
            .iter()
            .all(|v| v == 5.0)
    );
}

#[test]
fn integers_are_in_range_and_unbiased() {
    let a = Rng::seed(9).integers(-3, 4, &[70_000]);
    let mut counts = [0usize; 7];
    for v in a.iter() {
        assert!((-3..4).contains(&v));
        counts[(v + 3) as usize] += 1;
    }
    for (k, c) in counts.iter().enumerate() {
        assert!(
            (*c as f64 / 70_000.0 - 1.0 / 7.0).abs() < 0.01,
            "value {k}: {c}"
        );
    }
    // A range of one value, and the extremes of i64.
    assert!(Rng::seed(1).integers(4, 5, &[10]).iter().all(|v| v == 4));
    let wide = Rng::seed(1).integers(i64::MIN, i64::MAX, &[100]);
    assert_eq!(wide.size(), 100);
}

#[test]
fn choice_without_replacement_has_no_repeats() {
    let mut r = Rng::seed(11);
    let pool = np::arange(0, 50, 1);
    for n in [0, 1, 10, 25, 26, 50] {
        let picked = r.choice(&pool, n, false);
        assert_eq!(picked.shape(), &[n]);
        let mut v = picked.to_vec();
        v.sort();
        v.dedup();
        assert_eq!(v.len(), n, "n={n}");
        assert!(v.iter().all(|x| (0..50).contains(x)));
    }
    let with = r.choice(&np::array![1.5, 2.5], 1000, true);
    assert_eq!(with.shape(), &[1000]);
    let ones = with.iter().filter(|&v| v == 1.5).count();
    assert!(
        (400..600).contains(&ones),
        "replacement is roughly fair: {ones}"
    );
}

#[test]
fn choice_picks_rows_of_a_matrix() {
    let m = np::arange(0, 12, 1).reshape(&[4, 3]);
    let picked = Rng::seed(2).choice(&m, 3, false);
    assert_eq!(picked.shape(), &[3, 3]);
    for i in 0..3 {
        let row = picked.row(i).to_vec();
        assert_eq!(row[1], row[0] + 1);
        assert_eq!(row[2], row[0] + 2);
        assert_eq!(row[0] % 3, 0, "whole rows are kept intact");
    }
}

#[test]
fn shuffle_and_permutation_are_permutations() {
    let mut r = Rng::seed(13);
    let mut a = np::arange(0, 100, 1);
    r.shuffle(&mut a);
    assert_ne!(
        a,
        np::arange(0, 100, 1),
        "100 elements staying in order is impossible"
    );
    let mut v = a.to_vec();
    v.sort();
    assert_eq!(v, (0..100).collect::<Vec<_>>());

    // 2-d arrays are shuffled along the first axis; rows stay intact.
    let mut m = np::arange(0, 60, 1).reshape(&[20, 3]);
    r.shuffle(&mut m);
    assert_eq!(m.shape(), &[20, 3]);
    let mut firsts: Vec<i32> = (0..20).map(|i| m.get(&[i, 0])).collect();
    for i in 0..20 {
        assert_eq!(m.get(&[i, 1]), m.get(&[i, 0]) + 1);
        assert_eq!(m.get(&[i, 2]), m.get(&[i, 0]) + 2);
    }
    firsts.sort();
    assert_eq!(firsts, (0..20).map(|i| i * 3).collect::<Vec<_>>());

    let p = r.permutation(0);
    assert_eq!(p.size(), 0);
    let mut p = r.permutation(30).to_vec();
    p.sort();
    assert_eq!(p, (0..30).collect::<Vec<_>>());
}

#[test]
fn permutation_positions_are_roughly_uniform() {
    // Element 0 should land at each of 4 positions about a quarter of the time.
    let mut r = Rng::seed(17);
    let mut at = [0usize; 4];
    for _ in 0..20_000 {
        let p = r.permutation(4).to_vec();
        at[p.iter().position(|&x| x == 0).unwrap()] += 1;
    }
    for c in at {
        assert!((c as f64 / 20_000.0 - 0.25).abs() < 0.015, "{at:?}");
    }
}

#[test]
fn discrete_distributions_have_the_right_moments() {
    let n = 100_000usize;
    let to_f = |a: np::Array<i64>| a.iter().map(|v| v as f64).collect::<Vec<_>>();

    let b = to_f(Rng::seed(21).binomial(10, 0.3, &[n]));
    let (m, s) = mean_std(&b);
    assert!(
        (m - 3.0).abs() < 0.03 && (s * s - 2.1).abs() < 0.06,
        "m={m} v={}",
        s * s
    );
    assert!(b.iter().all(|&k| (0.0..=10.0).contains(&k)));

    // Large n goes through the exact splitting path.
    let b = to_f(Rng::seed(22).binomial(5000, 0.2, &[n]));
    let (m, s) = mean_std(&b);
    assert!((m - 1000.0).abs() < 0.5, "m={m}");
    assert!((s * s / 800.0 - 1.0).abs() < 0.03, "v={}", s * s);

    assert!(Rng::seed(1).binomial(7, 0.0, &[5]).iter().all(|k| k == 0));
    assert!(Rng::seed(1).binomial(7, 1.0, &[5]).iter().all(|k| k == 7));
    assert!(Rng::seed(1).binomial(0, 0.5, &[5]).iter().all(|k| k == 0));

    for lam in [0.5, 4.0, 60.0, 2500.0] {
        let p = to_f(Rng::seed(23).poisson(lam, &[n]));
        let (m, s) = mean_std(&p);
        let tol = 5.0 * (lam / n as f64).sqrt();
        assert!((m - lam).abs() < tol, "lam={lam} mean={m}");
        assert!((s * s / lam - 1.0).abs() < 0.04, "lam={lam} var={}", s * s);
        assert!(p.iter().all(|&k| k >= 0.0));
    }
    assert!(Rng::seed(1).poisson(0.0, &[5]).iter().all(|k| k == 0));
}

#[test]
fn continuous_distributions_have_the_right_moments() {
    let n = 100_000usize;
    let e = Rng::seed(31).exponential(2.0, &[n]).to_vec();
    let (m, s) = mean_std(&e);
    assert!((m - 2.0).abs() < 0.03 && (s - 2.0).abs() < 0.05);
    assert!(e.iter().all(|&v| v >= 0.0));

    for (k, theta) in [(0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (9.0, 0.5)] {
        let g = Rng::seed(32).gamma(k, theta, &[n]).to_vec();
        let (m, s) = mean_std(&g);
        let (em, ev) = (k * theta, k * theta * theta);
        assert!(
            (m - em).abs() < 0.03 * ev.sqrt() * 3.0,
            "gamma({k},{theta}) mean {m}"
        );
        assert!(
            (s * s / ev - 1.0).abs() < 0.06,
            "gamma({k},{theta}) var {}",
            s * s
        );
        assert!(g.iter().all(|&v| v >= 0.0));
    }

    let b = Rng::seed(33).beta(2.0, 5.0, &[n]).to_vec();
    let (m, s) = mean_std(&b);
    assert!((m - 2.0 / 7.0).abs() < 0.005);
    let ev = 2.0 * 5.0 / (49.0 * 8.0);
    assert!((s * s - ev).abs() < 0.002);
    assert!(b.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
fn invalid_parameters_panic() {
    let panics = |f: fn()| std::panic::catch_unwind(f).is_err();
    assert!(panics(|| {
        Rng::seed(1).integers(3, 3, &[1]);
    }));
    assert!(panics(|| {
        Rng::seed(1).normal(0.0, -1.0, &[1]);
    }));
    assert!(panics(|| {
        Rng::seed(1).binomial(5, 1.5, &[1]);
    }));
    assert!(panics(|| {
        Rng::seed(1).binomial(5, f64::NAN, &[1]);
    }));
    assert!(panics(|| {
        Rng::seed(1).poisson(-1.0, &[1]);
    }));
    assert!(panics(|| {
        Rng::seed(1).exponential(-1.0, &[1]);
    }));
    assert!(panics(|| {
        Rng::seed(1).beta(0.0, 1.0, &[1]);
    }));
    assert!(panics(|| {
        Rng::seed(1).gamma(-1.0, 1.0, &[1]);
    }));
    assert!(panics(|| {
        Rng::seed(1).choice(&np::arange(0, 3, 1), 4, false);
    }));
    assert!(panics(|| {
        Rng::seed(1).choice(&np::zeros::<f64>(&[0]), 1, true);
    }));
    assert!(panics(|| {
        Rng::seed(1).shuffle(&mut np::Array::scalar(1.0));
    }));
}

#[test]
fn works_with_every_element_type_in_choice_and_shuffle() {
    let mut r = Rng::seed(4);
    let f = r.choice(&np::array![1.0_f32, 2.0, 3.0], 2, false);
    let b = r.choice(&np::array![true, false], 5, true);
    let u = r.choice(&np::array![1_u8, 2, 3], 3, false);
    assert_eq!((f.size(), b.size(), u.size()), (2, 5, 3));
    let mut s = np::array![1_usize, 2, 3, 4];
    r.shuffle(&mut s);
    assert_eq!(s.sum(), 10);
}
