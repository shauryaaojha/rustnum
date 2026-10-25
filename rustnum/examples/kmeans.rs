//! k-means clustering of three Gaussian blobs, using a broadcasting distance matrix.
//!
//! Run with: `cargo run --release -p rustnum --example kmeans`
//!
//! The equivalent NumPy code:
//!
//! ```python
//! import numpy as np
//!
//! rng = np.random.default_rng(3)
//! centers = np.array([[0.0, 0.0], [6.0, 6.0], [0.0, 7.0]])
//! x = np.concatenate([rng.normal(0, 0.6, (100, 2)) + c for c in centers])
//!
//! k = 3
//! centroids = x[[0, 100, 200]]                 # one starting point per blob
//! for _ in range(100):
//!     diff = x[:, None, :] - centroids[None, :, :]       # (n, k, 2)
//!     d2 = (diff ** 2).sum(axis=2)                       # (n, k) squared distances
//!     labels = d2.argmin(axis=1)                         # nearest centroid
//!     onehot = (labels[:, None] == np.arange(k)).astype(float)   # (n, k)
//!     new = (onehot.T @ x) / onehot.sum(axis=0)[:, None]         # cluster means
//!     if np.allclose(new, centroids, atol=1e-9):
//!         break
//!     centroids = new
//!
//! print(centroids)
//! ```
//!
//! The starting centroids are three data points, one from each blob (the data is
//! generated blob by blob, so rows 0, 100 and 200 are known to be in different
//! blobs). A real k-means uses random restarts or k-means++ instead.

use rustnum as np;

fn main() {
    let mut rng = np::random::Rng::seed(3);

    // Three blobs of 100 points each, around three centers.
    let blob0 = rng.normal(0.0, 0.6, &[100, 2]) + np::array![0.0, 0.0];
    let blob1 = rng.normal(0.0, 0.6, &[100, 2]) + np::array![6.0, 6.0];
    let blob2 = rng.normal(0.0, 0.6, &[100, 2]) + np::array![0.0, 7.0];
    let x = np::concatenate(&[&blob0, &blob1, &blob2], 0);
    println!("data shape = {:?}", x.shape());

    let k: usize = 3;
    let mut centroids = x.take(&[0, 100, 200], 0);
    let cluster_ids = np::arange(0, k, 1); // [0, 1, 2] as usize

    for step in 0..100 {
        // Squared distance from every point to every centroid.
        // [n, 1, 2] - [1, k, 2] broadcasts to [n, k, 2]; sum the last axis to get [n, k].
        let diff = x.expand_dims(1) - centroids.expand_dims(0);
        let d2 = diff.square().sum_axis(2);

        // Nearest centroid for each point: shape [n], values in 0..k.
        let labels = d2.argmin_axis(1);

        // One-hot matrix [n, k]: 1.0 where the point belongs to the cluster.
        let onehot = labels.expand_dims(1).eq_elem(&cluster_ids).astype::<f64>();

        // New centroid = mean of the points in each cluster.
        let counts = onehot.sum_axis(0); // [k]
        let sums = onehot.t().matmul(&x); // [k, 2]
        let updated = sums / counts.expand_dims(1);

        let done = updated.allclose(&centroids, 0.0, 1e-9);
        centroids = updated;
        if done {
            println!("converged after {} steps", step + 1);
            println!("points per cluster = {counts}");
            break;
        }
    }

    println!("centroids =\n{centroids}");
    println!("true centers: [0, 0], [6, 6], [0, 7]");

    // Each centroid should be close to the center its blob was drawn around.
    let truth = np::array![[0.0, 0.0], [6.0, 6.0], [0.0, 7.0]];
    assert!(
        centroids.allclose(&truth, 0.0, 0.3),
        "centroids are far from the true centers"
    );
}
