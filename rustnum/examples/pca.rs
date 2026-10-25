//! Principal component analysis (PCA) on random 2-D correlated data.
//!
//! Run with: `cargo run --release -p rustnum --example pca`
//!
//! The equivalent NumPy code:
//!
//! ```python
//! import numpy as np
//!
//! rng = np.random.default_rng(1)
//! n = 500
//! z = rng.standard_normal((n, 2))
//! mix = np.array([[3.0, 1.0], [0.0, 1.0]])
//! x = z @ mix                                # correlated columns
//!
//! xc = x - x.mean(axis=0)                    # center
//! cov = xc.T @ xc / (n - 1)                  # covariance matrix
//! vals, vecs = np.linalg.eigh(cov)           # ascending eigenvalues
//! vals, vecs = vals[::-1], vecs[:, ::-1]     # largest first
//! ratio = vals / vals.sum()                  # explained variance ratio
//! scores = xc @ vecs                         # project onto the components
//!
//! print(ratio)
//! ```
//!
//! For this mixing matrix the covariance is close to `[[9, 3], [3, 2]]`, so the
//! explained variance ratio is close to `[0.92, 0.08]`.

use rustnum as np;

fn main() {
    let n: usize = 500;
    let mut rng = np::random::Rng::seed(1);

    // Correlated 2-D data: x = z @ mix, where z is standard normal.
    let z = rng.standard_normal(&[n, 2]);
    let mix = np::array![[3.0, 1.0], [0.0, 1.0]];
    let x = z.matmul(&mix);

    // Center each column, then build the covariance matrix.
    let mean = x.mean_axis(0);
    let xc = &x - &mean;
    let cov = xc.t().matmul(&xc) / (n as f64 - 1.0);
    println!("mean       = {mean}");
    println!("covariance =\n{cov}");

    // eigh returns ascending eigenvalues; flip so the largest comes first.
    let (vals, vecs) = np::linalg::eigh(&cov);
    let vals = vals.flip(0);
    let vecs = vecs.flip(1);
    println!("eigenvalues  = {vals}");
    println!("components (columns) =\n{vecs}");

    // Explained variance ratio.
    let ratio = &vals / vals.sum();
    println!("explained variance ratio = {ratio}");

    // Project onto the principal components.
    let scores = xc.matmul(&vecs);
    println!("scores shape = {:?}", scores.shape());

    // The scores are uncorrelated: their covariance matrix is diagonal,
    // and the diagonal holds the eigenvalues.
    let score_cov = scores.t().matmul(&scores) / (n as f64 - 1.0);
    println!("covariance of scores =\n{score_cov}");

    assert!((ratio.sum() - 1.0).abs() < 1e-9);
    assert!(ratio.get(&[0]) > 0.85);
    assert!(score_cov.get(&[0, 1]).abs() < 1e-8);
}
