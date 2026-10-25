//! Linear regression two ways: normal equations and gradient descent.
//!
//! Run with: `cargo run --release -p rustnum --example linear_regression`
//!
//! The equivalent NumPy code:
//!
//! ```python
//! import numpy as np
//!
//! rng = np.random.default_rng(0)
//! n = 200
//! x = rng.normal(0, 1, (n, 3))
//! true_w = np.array([2.0, -1.0, 0.5])
//! y = x @ true_w + 4.0 + rng.normal(0, 0.1, n)
//!
//! xb = np.concatenate([x, np.ones((n, 1))], axis=1)   # add a bias column
//!
//! # 1. Normal equations: (X^T X) w = X^T y
//! w_ne = np.linalg.solve(xb.T @ xb, xb.T @ y)
//!
//! # 2. Gradient descent on the mean squared error
//! w_gd = np.zeros(4)
//! for _ in range(500):
//!     grad = (2.0 / n) * xb.T @ (xb @ w_gd - y)
//!     w_gd -= 0.1 * grad
//!
//! print(w_ne, w_gd)
//! assert np.allclose(w_ne, w_gd, atol=1e-3)
//! ```
//!
//! The random numbers differ from NumPy's, so the fitted weights are close to
//! `[2.0, -1.0, 0.5, 4.0]` but not identical to the NumPy run.

use rustnum as np;

fn main() -> np::Result<()> {
    let n: usize = 200;
    let mut rng = np::random::Rng::seed(0);

    // Synthetic data: y = x @ true_w + 4.0 + noise.
    let x = rng.normal(0.0, 1.0, &[n, 3]);
    let true_w = np::array![2.0, -1.0, 0.5];
    let noise = rng.normal(0.0, 0.1, &[n]);
    let y = x.matmul(&true_w) + 4.0 + &noise;

    // Add a column of ones so the last weight is the intercept.
    let ones = np::ones::<f64>(&[n, 1]);
    let xb = np::concatenate(&[&x, &ones], 1);
    let xt = xb.t();

    // 1. Normal equations: (X^T X) w = X^T y.
    let xtx = xt.matmul(&xb);
    let xty = xt.matmul(&y);
    let w_ne = np::linalg::solve(&xtx, &xty)?;

    // 2. Gradient descent on the mean squared error.
    let lr: f64 = 0.1;
    let scale = 2.0 / n as f64;
    let mut w_gd = np::zeros::<f64>(&[4]);
    for _ in 0..500 {
        let err = xb.matmul(&w_gd) - &y;
        let grad = xt.matmul(&err) * scale;
        w_gd -= grad * lr;
    }

    println!("true weights          = [2, -1, 0.5, 4]");
    println!("normal equations      = {w_ne}");
    println!("gradient descent      = {w_gd}");

    let max_diff = (&w_ne - &w_gd).abs().max();
    println!("max |w_ne - w_gd|     = {max_diff:e}");
    assert!(w_ne.allclose(&w_gd, 0.0, 1e-3), "the two fits disagree");

    // Mean squared error of the normal-equation fit.
    let resid = xb.matmul(&w_ne) - &y;
    println!("mean squared error    = {:.5}", resid.square().mean());
    Ok(())
}
